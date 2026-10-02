#!/usr/bin/env python3
"""Opt-in EPUB -> reviewed performance script -> provider-neutral TTS plan."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.orchestrator.services.audiobook_script import (
    SYSTEM, analyze, build_plan, prepare, read_epub, save, verify_script, select_sample, document_digest,
)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    p.add_argument('epub'); p.add_argument('output'); p.add_argument('--batch-chars',type=int,default=6000)
    p.add_argument('--context-blocks',type=int,default=8)
    p.add_argument('--sample-range',action='append',help='Zero-based START:END, repeatable; END exclusive')
    p = sub.add_parser('analyze')
    p.add_argument('directory'); p.add_argument('--batch-chars',type=int)
    p.add_argument('--context-blocks',type=int)
    p.add_argument('--responses',help='Directory with batch-00000.json etc.; no API calls')
    p = sub.add_parser('plan')
    p.add_argument('directory'); p.add_argument('voices'); p.add_argument('--approvals')
    p.add_argument('--max-chars',type=int,default=200)
    p.add_argument('--unresolved-policy',choices=['block','narrator'],default='block')
    p.add_argument('--decisions',help='Content-bound explicit narrator fallback decisions')
    args = parser.parse_args()
    if args.command == 'prepare':
        directory = Path(args.output)
        doc = read_epub(args.epub)
        if args.sample_range:
            ranges = [tuple(map(int,r.split(':'))) for r in args.sample_range]
            if any(len(r)!=2 for r in ranges):
                raise ValueError('Sample range must be START:END')
            doc = select_sample(doc,ranges)
        batches = prepare(doc,args.batch_chars,args.context_blocks)
        for stale in ('script.json', 'tts_plan.json'):
            (directory/stale).unlink(missing_ok=True)
        save(directory/'source.json',doc)
        save(directory/'requests.json',{'system':SYSTEM,'batches':batches,
             'document_digest':document_digest(doc),
             'config':{'batch_chars':args.batch_chars,'context_blocks':args.context_blocks}})
        print(f"Extracted {len(doc['blocks'])} blocks; {len(doc['skipped'])} excluded spine entries")
    elif args.command == 'analyze':
        directory = Path(args.directory)
        for stale in ('script.json', 'tts_plan.json'):
            (directory/stale).unlink(missing_ok=True)
        kwargs = {}
        doc = load(directory/'source.json')
        prepared = load(directory/'requests.json')
        if prepared.get('document_digest') != document_digest(doc) or prepared['system'] != SYSTEM:
            raise ValueError('Prepared source/prompt changed; prepare again')
        config = prepared['config']
        for key in ('batch_chars','context_blocks'):
            value = getattr(args,key)
            if value is not None and value != config[key]:
                raise ValueError('Batch configuration differs from prepare; prepare again')
        if args.responses:
            # Imported responses have their own cache namespace, never reuse API cache.
            responses = Path(args.responses)
            kwargs['use_cache'] = False
            def imported(payload):
                envelope = load(responses/(payload['batch_id']+'.json'))
                if envelope.get('request_digest') != payload['request_digest']:
                    raise ValueError('Imported response belongs to another request')
                return envelope['response']
            kwargs['model_call'] = imported
            kwargs['provenance'] = {'mode':'imported_response','model':None,
                                    'note':'No external model API was called by this command'}
            output = directory/'imported'
        else:
            output = directory/'api'
        script = analyze(doc,output,**config,**kwargs)
        save(directory/'script.json',script)
        print(f"Annotated {len(script['segments'])} segments; review {sum(s['needs_review'] for s in script['segments'])}")
    else:
        directory = Path(args.directory)
        # Remove stale runnable plan before validating edited inputs.
        (directory/'tts_plan.json').unlink(missing_ok=True)
        script = load(directory/'script.json')
        plan = build_plan(script,load(args.voices),load(args.approvals) if args.approvals else None,args.max_chars,
                          document=load(directory/'source.json'),unresolved_policy=args.unresolved_policy,
                          decisions=load(args.decisions) if args.decisions else None)
        save(directory/'tts_plan.json',plan)
        print(f"Ready={plan['ready']}; jobs={len(plan['jobs'])}; blockers={len(plan['blockers'])}")
        return 0 if plan['ready'] else 2
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Failed: {exc}",file=sys.stderr)
        sys.exit(1)
