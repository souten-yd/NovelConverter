"""Offline acceptance tests; no copyrighted source or external API needed."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from app.orchestrator.services.audiobook_script import (
    read_epub, prepare, validate_response, analyze, build_plan, verify_script, digest, document_digest, select_sample, split_spans,
)


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.doc = {'version':1,'source_sha256':'abc','blocks':[
            {'id':'b0','spine_index':0,'text':'「姉さん」妹が言った。','ruby':[]},
            {'id':'b1','spine_index':0,'text':'「え」','ruby':[]} ]}
        self.characters = [{'id':'sister','name':'妹','aliases':[]}]
        def span(start,end,kind,speaker,confidence=.95):
            return dict(start=start,end=end,kind=kind,speaker_id=speaker,confidence=confidence,
                        emotion='neutral',delivery='neutral',needs_review=False,
                        evidence=[{'block_id':'b0','quote':'妹が言った。'}])
        self.response = {'characters':self.characters,'annotations':[
            {'block_id':'b0','segments':[span(0,5,'dialogue','sister'),span(5,11,'narration','narrator')]},
            {'block_id':'b1','segments':[span(0,3,'dialogue','unknown',.5)]}]}
        self.req = prepare(self.doc)[0]

    def test_epub_spine_ruby_inline_and_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.epub'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('META-INF/container.xml','<container><rootfile full-path="OPS/book.opf"/></container>')
                z.writestr('OPS/book.opf','<package><manifest><item id="a" href="a.xhtml"/><item id="b" href="b.xhtml"/></manifest><spine><itemref idref="b"/><itemref idref="a"/></spine></package>')
                z.writestr('OPS/a.xhtml','<html><head><title>DO NOT READ</title></head><body><p>後</p></body></html>')
                z.writestr('OPS/b.xhtml','<html><body><p>「<span>空</span><ruby><rb>色</rb><rt>いろ</rt></ruby>」</p></body></html>')
            doc=read_epub(path)
            self.assertEqual([b['text'] for b in doc['blocks']],['「空色」','後'])
            self.assertEqual(doc['blocks'][0]['ruby'],[{'base':'色','reading':'いろ','start':2,'end':3}])

    def test_validation_and_unknown_gate(self):
        chars,segs=validate_response(self.req,self.response,[])
        self.assertEqual(segs[0]['speaker_id'],'sister')
        self.assertTrue(segs[-1]['needs_review'])
        script={'version':1,'document_digest':document_digest(self.doc),'source_sha256':'abc','characters':chars,'segments':segs}
        verify_script(self.doc,script)
        self.assertFalse(build_plan(script,{})['ready'])
        self.assertEqual(build_plan(script,{})['jobs'],[])

    def test_reject_bad_output(self):
        for mutation in [
            lambda r:r['annotations'].pop(),
            lambda r:r['annotations'].append(r['annotations'][0]),
            lambda r:r['annotations'][0]['segments'][0].update(end=6),
            lambda r:r['annotations'][0]['segments'][0].update(speaker_id='invented'),
            lambda r:r['annotations'][0]['segments'][0].update(confidence=float('nan')),
            lambda r:r['annotations'][0]['segments'][0].update(evidence=[{'block_id':'b0','quote':'嘘'}]),
        ]:
            r=copy.deepcopy(self.response);mutation(r)
            with self.assertRaises(ValueError):validate_response(self.req,r,[])

    def test_voice_identity_and_content_cache(self):
        r=copy.deepcopy(self.response)
        r['annotations'][1]['segments'][0].update(speaker_id='sister',kind='thought',confidence=.95)
        chars,segs=validate_response(self.req,r,[])
        script={'version':1,'document_digest':document_digest(self.doc),'source_sha256':'abc','characters':chars,'segments':segs}
        voices={c:{'provider':'future','voice_id':c,'revision':'1'} for c in ['sister','narrator']}
        plan=build_plan(script,voices,max_chars=2)
        self.assertTrue(plan['ready'])
        self.assertEqual(''.join(j['text'] for j in plan['jobs']), ''.join(b['text'] for b in self.doc['blocks']))
        self.assertEqual(plan['jobs'][0]['voice'],plan['jobs'][-1]['voice'])
        voices['sister']['revision']='2'
        self.assertNotEqual(plan['jobs'][0]['cache_key'],build_plan(script,voices,max_chars=2)['jobs'][0]['cache_key'])
        segs[0]['needs_review']=True
        self.assertFalse(build_plan(script,voices)['ready'])
        approvals={segs[0]['id']:digest(segs[0])}
        self.assertTrue(build_plan(script,voices,approvals)['ready'])
        segs[0]['emotion']='怒り'
        self.assertFalse(build_plan(script,voices,approvals)['ready'])

    def test_resume_and_import_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def model(payload):calls.append(payload);return self.response
            script=analyze(self.doc,tmp,model_call=model)
            analyze(self.doc,tmp,model_call=model)
            self.assertEqual(len(calls),1)
            analyze(self.doc,tmp,model_call=model,use_cache=False)
            self.assertEqual(len(calls),2)
            script['segments'][0]['text']='改変'
            with self.assertRaises(ValueError):verify_script(self.doc,script)

    def test_registry_alias_conflict_and_redefinition(self):
        with self.assertRaises(ValueError):
            validate_response(self.req,self.response,[{'id':'sister','name':'別人','aliases':[]}])
        chars,segs=validate_response(self.req,self.response,[])
        chars.append({'id':'other','name':'別人','aliases':['妹']})
        plan=build_plan({'source_sha256':'abc','characters':chars,'segments':segs},{})
        self.assertIn('ambiguous character alias: 妹',plan['blockers'])

    def test_whole_block_and_exact_text_spans(self):
        r=copy.deepcopy(self.response)
        a=r['annotations'][1]['segments'][0]
        a.pop('start');a.pop('end');a['whole_block']=True
        for a in r['annotations'][0]['segments']:
            a['text']=self.doc['blocks'][0]['text'][a.pop('start'):a.pop('end')]
        _,segs=validate_response(self.req,r,[])
        self.assertEqual(''.join(s['text'] for s in segs), '「姉さん」妹が言った。「え」')
        r['annotations'][0]['segments'][0]['text']='「兄さん」'
        with self.assertRaises(ValueError):validate_response(self.req,r,[])

    def test_silence_and_performance_review(self):
        d=copy.deepcopy(self.doc);d['blocks'][1]['text']='「──」'
        r=copy.deepcopy(self.response)
        s=r['annotations'][1]['segments'][0]
        s.update(end=4,speaker_id='sister',confidence=.99,emotion='怒り')
        _,segs=validate_response(prepare(d)[0],r,[])
        self.assertIn('possible_silence',segs[-1]['review_reasons'])
        self.assertIn('uncertain_emotion',segs[-1]['review_reasons'])

    def test_sample_scope_and_no_context_across_gaps(self):
        d=copy.deepcopy(self.doc)
        d['blocks'].append({'id':'b2','text':'後','spine_index':0,'ruby':[]})
        sample=select_sample(d,[(0,1),(2,3)])
        batches=prepare(sample)
        self.assertEqual(len(batches),2)
        self.assertTrue(all(not r['context'] for r in batches))
        with self.assertRaises(ValueError):select_sample(d,[(1,3),(0,1)])
        chars,segs=validate_response(self.req,self.response,[])
        script={'version':1,'source_sha256':'abc','document_digest':document_digest(self.doc),'characters':chars,'segments':segs}
        with self.assertRaises(ValueError):verify_script(d,script)

    def test_explicit_unknown_fallback_preserves_attribution(self):
        chars,segs=validate_response(self.req,self.response,[])
        script={'version':1,'source_sha256':'abc','document_digest':document_digest(self.doc),'characters':chars,'segments':segs}
        voices={c:{'provider':'evaluation','voice_id':c,'revision':'1'} for c in ['sister','narrator']}
        unknown=segs[-1]
        decisions={unknown['id']:{'action':'narrator_fallback','segment_digest':digest(unknown),'reason':'cannot disambiguate'}}
        self.assertFalse(build_plan(script,voices,decisions=decisions)['ready'])
        self.assertFalse(build_plan(script,voices,unresolved_policy='narrator')['ready'])
        plan=build_plan(script,voices,unresolved_policy='narrator',decisions=decisions,document=self.doc)
        self.assertTrue(plan['ready'])
        self.assertEqual(plan['jobs'][-1]['speaker_id'],'unknown')
        self.assertEqual(plan['jobs'][-1]['render_speaker_id'],'narrator')
        self.assertEqual(plan['jobs'][-1]['kind'],'dialogue')
        unknown['emotion']='悲しみ'
        self.assertFalse(build_plan(script,voices,unresolved_policy='narrator',decisions=decisions)['ready'])

    def test_ruby_offsets_splitting_and_voice_snapshot(self):
        import xml.etree.ElementTree as ET
        from app.orchestrator.services.audiobook_script import _text_and_ruby
        node=ET.fromstring('<p>　<ruby>今日<rt>きょう</rt></ruby>も、<ruby>今日<rt>こんにち</rt></ruby>。</p>')
        text,ruby=_text_and_ruby(node)
        self.assertEqual(text,'今日も、今日。')
        self.assertEqual([text[r['start']:r['end']] for r in ruby],['今日','今日'])
        self.assertEqual([r['start'] for r in ruby],[0,4])
        spans=list(split_spans(text,3,ruby))
        self.assertEqual(''.join(text[a:b] for a,b in spans),text)
        for a,b in spans:
            self.assertTrue(all(not r['start'] < b < r['end'] for r in ruby))
        with self.assertRaises(ValueError):list(split_spans(text,1,ruby))
        chars,segs=validate_response(self.req,self.response,[])
        script={'source_sha256':'abc','characters':chars,'segments':segs[:-1]}
        voices={c:{'provider':'test','voice_id':c,'revision':'1'} for c in ['sister','narrator']}
        p=build_plan(script,voices)
        voices['sister']['revision']='2'
        self.assertEqual(p['jobs'][0]['voice']['revision'],'1')

    def test_failed_rerun_removes_stale_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyze(self.doc,tmp,model_call=lambda p:self.response)
            self.assertTrue((Path(tmp)/'script.json').exists())
            with self.assertRaises(ValueError):
                analyze(self.doc,tmp,model_call=lambda p:{},use_cache=False)
            self.assertFalse((Path(tmp)/'script.json').exists())
            self.assertFalse((Path(tmp)/'review.json').exists())

    def test_cli_end_to_end_and_stale_import(self):
        import subprocess,sys
        from app.orchestrator.services.audiobook_script import SYSTEM,save
        cli=Path(__file__).resolve().parents[1]/'scripts/audiobook_script.py'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);responses=root/'responses'
            def run(*args):
                return subprocess.run([sys.executable,str(cli),*map(str,args)],capture_output=True,text=True)
            save(root/'source.json',self.doc)
            save(root/'requests.json',{'system':SYSTEM,'config':{'batch_chars':6000,'context_blocks':8},
                                     'document_digest':document_digest(self.doc),'batches':[self.req]})
            save(responses/'batch-00000.json',{'request_digest':self.req['request_digest'],'response':self.response})
            result=run('analyze',root,'--responses',responses)
            self.assertEqual(result.returncode,0,result.stderr)
            save(root/'voices.json',{c:{'provider':'test','voice_id':c,'revision':'1'} for c in ['sister','narrator']})
            result=run('plan',root,root/'voices.json')
            self.assertEqual(result.returncode,2,result.stderr)
            plan=json.loads((root/'tts_plan.json').read_text())
            self.assertFalse(plan['ready']);self.assertEqual(plan['jobs'],[])
            save(responses/'batch-00000.json',{'request_digest':'stale','response':self.response})
            result=run('analyze',root,'--responses',responses)
            self.assertEqual(result.returncode,1)
            self.assertFalse((root/'script.json').exists())
            self.assertFalse((root/'tts_plan.json').exists())
            result=run('analyze',root,'--responses',responses,'--batch-chars','1000')
            self.assertEqual(result.returncode,1)

if __name__=='__main__':unittest.main()
