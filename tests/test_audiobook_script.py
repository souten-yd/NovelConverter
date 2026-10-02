"""Offline acceptance tests; no copyrighted source or external API needed."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from app.orchestrator.services.audiobook_script import (
    read_epub, prepare, validate_response, analyze, build_plan, verify_script, digest,
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
            self.assertEqual(doc['blocks'][0]['ruby'],[{'base':'色','reading':'いろ'}])

    def test_validation_and_unknown_gate(self):
        chars,segs=validate_response(self.req,self.response,[])
        self.assertEqual(segs[0]['speaker_id'],'sister')
        self.assertTrue(segs[-1]['needs_review'])
        script={'version':1,'source_sha256':'abc','characters':chars,'segments':segs}
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
        script={'version':1,'source_sha256':'abc','characters':chars,'segments':segs}
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

if __name__=='__main__':unittest.main()
