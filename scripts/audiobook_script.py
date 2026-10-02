#!/usr/bin/env python3
"""Opt-in EPUB -> reviewed performance script -> provider-neutral TTS plan."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.orchestrator.services.audiobook_script import (
    SYSTEM, analyze, build_plan, prepare, read_epub, save, verify_script,
)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    p.add_argument('epub'); p.add_argument('output'); p.add_argument('--batch-chars',type=int,default=6000)
    p = sub.add_parser('analyze')
    p.add_argument('directory'); p.add_argument('--batch-chars',type=int,default=6000)
    p.add_argument('--responses',help='Directory with batch-00000.json etc.; no API calls')
    p = sub.add_parser('plan')
    p.add_argument('directory'); p.add_argument('voices'); p.add_argument('--approvals')
    p.add_argument('--max-chars',type=int,default=200)
    args = parser.parse_args()
    if args.command == 'prepare':
        directory = Path(args.output)
        doc = read_epub(args.epub)
        batches = prepare(doc,args.batch_chars)
        for stale in ('script.json', 'tts_plan.json'):
            (directory/stale).unlink(missing_ok=True)
        save(directory/'source.json',doc)
        save(directory/'requests.json',{'system':SYSTEM,'batches':batches})
        print(f"Extracted {len(doc['blocks'])} blocks; {len(doc['skipped'])} excluded spine entries")
    elif args.command == 'analyze':
        directory = Path(args.directory)
        for stale in ('script.json', 'tts_plan.json'):
            (directory/stale).unlink(missing_ok=True)
        kwargs = {}
        if args.responses:
            # Imported responses have their own cache namespace, never reuse API cache.
            responses = Path(args.responses)
            kwargs['use_cache'] = False
            kwargs['model_call'] = lambda payload: load(responses/(payload['batch_id']+'.json'))
            output = directory/'imported'
        else:
            output = directory/'api'
        script = analyze(load(directory/'source.json'),output,batch_chars=args.batch_chars,**kwargs)
        save(directory/'script.json',script)
        print(f"Annotated {len(script['segments'])} segments; review {sum(s['needs_review'] for s in script['segments'])}")
    else:
        directory = Path(args.directory)
        # Remove stale runnable plan before validating edited inputs.
        (directory/'tts_plan.json').unlink(missing_ok=True)
        script = load(directory/'script.json')
        verify_script(load(directory/'source.json'),script)
        plan = build_plan(script,load(args.voices),load(args.approvals) if args.approvals else None,args.max_chars)
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
