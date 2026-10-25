"""Frozen 100-per-task inference diagnostic; no training or baseline mutation.

Prepare inputs and scoring rubric before inference. Full materials stay server-side.
Semantic grades are a separate review, never manufactured from lexical matches.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def text_cases():
    facts = [("中国的首都是哪个城市？", "北京"), ("法国的首都是哪个城市？", "巴黎"),
             ("日本的首都是哪个城市？", "东京"), ("一小时有多少分钟？", "60分钟"),
             ("一千米等于多少米？", "1000米"), ("一个星期有多少天？", "7天"),
             ("三角形有几条边？", "3条"), ("正方形有几个直角？", "4个"),
             ("地球绕着哪颗恒星公转？", "太阳"), ("水的化学式是什么？", "H2O"),
             ("氧气的化学式是什么？", "O2"), ("英文单词cat对应的中文是什么？", "猫"),
             ("英文单词dog对应的中文是什么？", "狗"), ("英文单词apple对应的中文是什么？", "苹果"),
             ("通常说的春夏秋冬一共是几个季节？", "4个"), ("一天有多少小时？", "24小时"),
             ("二进制的10等于十进制的几？", "2"), ("十进制的一位数字中最大的是几？", "9"),
             ("Python列表的第一个元素通常用哪个索引访问？", "0"), ("太阳通常从东方还是西方升起？", "东方")]
    rows = []
    for prompt, answer in facts:
        rows.append({"category": "basic_knowledge", "prompt": prompt, "reference": answer, "strict_format": False})
    rng = random.Random(20261025)
    for i in range(20):
        a,b = rng.randrange(11,70),rng.randrange(2,30)
        op = ["+", "-", "×"][i % 3]
        answer = a+b if op == "+" else a-b if op == "-" else a*b
        rows.append({"category": "arithmetic", "prompt": f"请计算 {a} {op} {b} 的结果。", "reference": str(answer), "strict_format": False})
    objects = ["雨伞","钥匙","水杯","铅笔","书包","帽子","手机","毛巾","手套","眼镜"]
    for i in range(20):
        a,b = objects[i%10],objects[(i+3)%10]
        prompt = f"记录：小林把{a}放进红盒子，小周把{b}放进蓝盒子。红盒子里放的是什么？" if i<10 else f"记录：小林的号码是{31+i}，小周的号码是{81-i}。请问小周的号码是多少？"
        rows.append({"category": "context_retrieval", "prompt": prompt, "reference": a if i<10 else str(81-i), "strict_format": False})
    sentences = ["这家店的服务很贴心，我很满意。","电影无聊透了，浪费时间。","今天的旅行太棒了。","这个产品质量很差，让人失望。",
                 "老师讲得清楚，我很喜欢。","客服态度恶劣，体验糟糕。","这顿饭真好吃。","房间又脏又吵，我很不满。",
                 "新耳机声音清晰，物有所值。","软件总是崩溃，令人烦躁。","这本书很精彩，值得推荐。","快递把东西摔坏了，太生气了。",
                 "朋友的帮助让我很感动。","演出拖沓乏味，我很后悔。","这里风景优美，令人愉快。","椅子很不舒服，不想再买。",
                 "问题终于解决了，真开心。","这次经历很糟糕，难以接受。","工作人员耐心负责，值得表扬。","饭菜冷了还难吃，体验很差。"]
    for i,s in enumerate(sentences):
        rows.append({"category": "sentiment", "prompt": f"“{s}”表达的是正面还是负面情感？", "reference": "正面" if i%2==0 else "负面", "strict_format": False})
    for i in range(10):
        values=rng.sample(range(1,50),3)
        rows.append({"category":"instruction_format", "prompt":f"把{values}中的数字从小到大排序，只输出一个JSON数组，不要解释。", "reference":json.dumps(sorted(values)), "strict_format":True})
    for i in range(10):
        target=f"蓝色河岸 Q{37+i}"
        rows.append({"category":"instruction_format", "prompt":f"请原样复制下面的文本，只输出文本本身：\n{target}", "reference":target, "strict_format":True})
    assert len(rows)==100
    return [{"id":f"T{i+1:03d}","task":"text",**r} for i,r in enumerate(rows)]


def first_pair(conversations):
    if isinstance(conversations,str): conversations=json.loads(conversations)
    for i,m in enumerate(conversations):
        if m["role"]=="user" and i+1<len(conversations) and conversations[i+1]["role"]=="assistant":
            return m["content"],conversations[i+1]["content"]
    raise ValueError("No direct user/assistant pair")


def prepare(root, run, out):
    import pyarrow.parquet as pq
    from PIL import Image
    from minimind_lab.data.video import qivd_split_indices
    from minimind_lab.training.evidence import RunEvidence

    evidence=RunEvidence(root,run.name);evidence.assert_sources_unchanged()
    out.mkdir(parents=True,exist_ok=False)
    (out/"inputs").mkdir()
    ledger=[json.loads(x) for x in (run/"events.jsonl").read_text().splitlines()]
    checkpoints={}
    for task,stage,name in [("text","llm-sft","llm-64m-sft-cuda.pt"),("image","vlm-sft","vlm-sft-cuda.pt"),("video","video-omni-sft","video-omni-sft-cuda.pt")]:
        assert any(e.get("kind")=="stage_completed" and e.get("stage")==stage for e in ledger)
        entries=[a for e in ledger if e.get("kind")=="artifacts_archived" and e.get("stage")==stage for a in e["artifacts"] if Path(a["archive"]).name==name]
        assert len(entries)==1
        a=entries[0];p=(root/a["archive"]).resolve();assert p.is_relative_to(run/"stages"/stage) and sha(p)==a["sha256"]
        checkpoints[task]={"stage":stage,"path":a["archive"],"sha256":a["sha256"]}
    cases=text_cases()
    pf=pq.ParquetFile(root/'data/raw/sft_i2t.parquet');valid=[]
    starts=[];start=0
    for g in range(pf.num_row_groups):starts.append(start);start+=pf.metadata.row_group(g).num_rows
    for g in range(pf.num_row_groups-1,-1,-1):
        rows=pf.read_row_group(g).to_pylist()
        for i in range(len(rows)-1,-1,-1):
            row=rows[i];c=json.loads(row['conversations']) if isinstance(row['conversations'],str) else row['conversations']
            if sum(str(m.get('content','')).count('<image>') for m in c if m['role']!='system')==1:
                q,a=first_pair(c);valid.append({"raw_row_index":starts[g]+i,"prompt":q,"reference":a,"bytes":row['image_bytes']})
            if len(valid)==512:break
        if len(valid)==512:break
    assert len(valid)==512
    unique={}
    for row in reversed(valid):
        key=hashlib.sha256(row['bytes']).hexdigest();old=unique.get(key)
        chinese=any('\u4e00'<=c<='\u9fff' for c in row['prompt'])
        if old is None or (chinese and not old['chinese']):unique[key]={**row,"chinese":chinese}
    selected=sorted(unique,key=lambda k:hashlib.sha256(f'20261025:{k}'.encode()).hexdigest())[:100]
    assert len(selected)==100
    for i,key in enumerate(selected):
        row=unique[key];relative=f"inputs/I{i+1:03d}.png";Image.open(io.BytesIO(row['bytes'])).convert('RGB').save(out/relative)
        cases.append({"id":f"I{i+1:03d}","task":"image","category":"natural_image_qa","prompt":row['prompt'],"reference":row['reference'],
                      "raw_row_index":row['raw_row_index'],"original_image_sha256":key,"input_image":relative,"input_sha256":sha(out/relative)})
    vr=root/'data/raw/qivd';rows=pq.read_table(vr/'metadata.parquet').to_pylist();indices=qivd_split_indices(rows,'test',48,2400,250)
    indices=sorted(indices,key=lambda i:hashlib.sha256(f"20261025:{rows[i]['id']}".encode()).hexdigest())[:100]
    for i,index in enumerate(indices):
        row=rows[index];video=vr/row['video_file_name']
        cases.append({"id":f"V{i+1:03d}","task":"video","category":row['category'],"prompt":row['question'],"reference":row['answer'],
                      "short_answer":row['short_answer'],"source_id":row['id'],"input_video":str(video.relative_to(root)),"input_sha256":sha(video)})
    protocol={"schema":1,"eval_id":out.name,"run_id":run.name,"created_at_utc":datetime.now(UTC).isoformat(),"checkpoints":checkpoints,
              "generation":{"precision":"float32","max_new_tokens":128,"temperature":0,"stop_at_eos":True,"seed":20261025,"batch_size":1,"kv_cache":False},
              "selection":{"text":"20 each: basic knowledge, arithmetic, context retrieval, sentiment, instruction format; assistant-authored diagnostics, not independent benchmark",
                           "image":"Last 512 valid single-marker SFT validation rows; deduplicate image byte SHA, prefer Chinese question per image, SHA-seed order, 100 distinct images",
                           "video":"QIVD test 250 rows, hash-seed order, first 100; overlaps earlier observed fixed test, not an untouched benchmark","image_pool_distinct":len(unique)},
              "scoring":{"primary":"Single project-assistant semantic review against references and visual evidence: correct=fully answers core request without material contradiction; incorrect=wrong, critical omission, nonanswer or explicit format violation; uncertain=reference/input insufficient. No answer-dependent resampling.",
                         "partial":"Count partial answers as incorrect, but retain explanatory notes","rates":"correct/100 and correct/(100-uncertain); uncertain count shown, not silently excluded",
                         "secondary":"Normalized exact reference match; lexical diagnostic only, not semantic correctness","judge_limit":"Single AI reviewer, not independent human annotation or multi-judge validated benchmark"},
              "evaluator_sha256":sha(__file__),"cases":cases}
    write(out/'protocol.json',protocol);shutil.copy2(__file__,out/'evaluator.py')
    manifest={"status":"prepared","eval_id":out.name,"run_id":run.name,"protocol_sha256":sha(out/'protocol.json'),"evaluator_sha256":sha(__file__),"case_counts":dict(Counter(c['task'] for c in cases))}
    write(out/'manifest.json',manifest)
    evidence.event('three_task_100_protocol_frozen',eval_id=out.name,protocol_sha256=manifest['protocol_sha256'],counts=manifest['case_counts'])
    print(json.dumps(manifest),flush=True)


def infer(root,run,out):
    import torch,yaml
    from PIL import Image,ImageDraw
    from transformers import AutoTokenizer,SiglipImageProcessor,SiglipVisionModel
    from minimind_lab.llm import MiniMindConfig,MiniMindForCausalLM
    from minimind_lab.vlm import MiniMindVLM,VLMConfig
    from minimind_lab.video import MiniMindVideoOmni,VideoOmniConfig
    from minimind_lab.data.video import decode_uniform_video
    from minimind_lab.training.evidence import RunEvidence

    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1','Only authorized GPU1'
    e=RunEvidence(root,run.name);e.assert_sources_unchanged()
    m=json.loads((out/'manifest.json').read_text());assert m['status']=='prepared' and m['protocol_sha256']==sha(out/'protocol.json') and m['evaluator_sha256']==sha(__file__)
    p=json.loads((out/'protocol.json').read_text());torch.manual_seed(20261025)
    tokenizer=AutoTokenizer.from_pretrained(root/'assets/tokenizer',local_files_only=True)
    cfgs={t:yaml.safe_load((run/'configs'/name).read_text()) for t,name in [('text','llm/pretrain-cuda.yaml'),('image','vlm/sft-cuda.yaml'),('video','video/sft-cuda.yaml')]}
    path=out/'predictions.jsonl';assert not path.exists();m.update(status='running',started_at_utc=datetime.now(UTC).isoformat());write(out/'manifest.json',m)
    results=[]
    try:
        for task in ['text','image','video']:
            spec=p['checkpoints'][task];checkpoint=root/spec['path'];assert sha(checkpoint)==spec['sha256']
            values=dict(cfgs[task]['model']);values.pop('language_checkpoint',None);values.pop('alignment_checkpoint',None)
            if task=='text':model=MiniMindForCausalLM(MiniMindConfig(**values));processor=None
            else:
                vision_path=root/values.pop('vision_encoder');processor=SiglipImageProcessor.from_pretrained(vision_path,local_files_only=True);vision=SiglipVisionModel.from_pretrained(vision_path,local_files_only=True)
                model=MiniMindVLM(VLMConfig(**values),vision) if task=='image' else MiniMindVideoOmni(VideoOmniConfig(**values),vision)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
            loaded=model.load_state_dict(state,strict=(task=='text'))
            if task!='text':assert not loaded.unexpected_keys and all(k.startswith('vision_encoder.') for k in loaded.missing_keys)
            del state
            model=model.to('cuda').eval()
            for case in [c for c in p['cases'] if c['task']==task]:
                prompt=case['prompt'];features=None
                if task=='image':
                    image_path=out/case['input_image'];assert sha(image_path)==case['input_sha256']
                    pixels=processor(images=[Image.open(image_path).convert('RGB')],return_tensors='pt')['pixel_values'].unsqueeze(0).to('cuda')
                    prompt=prompt.replace('<image>','<|image_pad|>'*64)
                elif task=='video':
                    video=root/case['input_video'];assert sha(video)==case['input_sha256'];frames=decode_uniform_video(video,8)
                    frame_dir=out/'inputs'/case['id'];frame_dir.mkdir();strip=Image.new('RGB',(1024,544),'white');draw=ImageDraw.Draw(strip)
                    for i,frame in enumerate(frames):
                        frame.save(frame_dir/f'frame-{i}.png');thumb=frame.copy();thumb.thumbnail((256,256));strip.paste(thumb,((i%4)*256,(i//4)*272));draw.text(((i%4)*256,(i//4)*272+256),f'Sampled input frame {i}',fill='black')
                    strip.save(frame_dir/'input-8-frames.png')
                    pixels=processor(images=frames,return_tensors='pt')['pixel_values'].unsqueeze(0).to('cuda');prompt='<|video_pad|>'*16+'\n'+prompt
                rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
                ids=tokenizer(rendered).input_ids;inputs=torch.tensor([ids],device='cuda');completion=[]
                assert len(ids)+128<=model.config.max_position_embeddings
                with torch.inference_mode():
                    if task=='image':features=model.encode_images(pixels)
                    elif task=='video':features=model.encode_video(pixels)
                    torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(128):
                        logits=(model(inputs) if task=='text' else model.forward_with_features(inputs,features))['logits'][0,-1]
                        token=int(logits.argmax().item());completion.append(token)
                        if token==tokenizer.eos_token_id:break
                        inputs=torch.cat([inputs,torch.tensor([[token]],device='cuda')],dim=1)
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                answer=tokenizer.decode(completion,skip_special_tokens=True)
                result={'id':case['id'],'task':task,'prompt':case['prompt'],'reference':case['reference'],'completion':answer,'prompt_token_ids':ids,'completion_token_ids':completion,'stopped_at_eos':completion[-1]==tokenizer.eos_token_id,'generation_seconds_excludes_visual_encoder':seconds}
                with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
                results.append(result);print(json.dumps({'id':case['id'],'tokens':len(completion),'eos':result['stopped_at_eos']}),flush=True)
                del inputs,logits,features
                if task!='text':del pixels
            del model
            if task!='text':del vision
            torch.cuda.empty_cache()
        assert len(results)==300 and len({r['id'] for r in results})==300
        e.assert_sources_unchanged();assert sha(out/'protocol.json')==m['protocol_sha256']
        m.update(status='completed',completed_at_utc=datetime.now(UTC).isoformat(),predictions_sha256=sha(path),predictions=300);write(out/'manifest.json',m)
        e.event('three_task_100_inference_completed',eval_id=out.name,manifest_sha256=sha(out/'manifest.json'),generated_cases=300)
    except Exception as error:
        m.update(status='failed',error_type=type(error).__name__,error=str(error));write(out/'manifest.json',m);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','infer']);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--eval-id',required=True);args=parser.parse_args()
    assert args.eval_id and all(c.isalnum() or c in '-_' for c in args.eval_id)
    run=args.run_dir.resolve();root=run.parents[2];sys.path.insert(0,str(root/'src'));out=run/'supplemental-evaluations'/args.eval_id
    prepare(root,run,out) if args.action=='prepare' else infer(root,run,out)
