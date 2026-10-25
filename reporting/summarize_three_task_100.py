"""Validate frozen inference + explicit single-reviewer labels; build private gallery.

Full gallery/predictions/reference data are not for public GitHub publication.
Only aggregate summary and condensed review labels should be published.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter,defaultdict
from datetime import UTC,datetime
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(re.findall(r'\w+',text.casefold()))


def summary_counts(verdicts):
    values=Counter(verdicts)
    if set(values)-{'correct','incorrect','uncertain'}:
        raise ValueError('Unknown review verdict')
    n=sum(values.values());eligible=n-values['uncertain']
    return {'samples':n,'correct':values['correct'],'incorrect':values['incorrect'],'uncertain':values['uncertain'],
            'correct_over_all':values['correct']/n if n else None,
            'assessable_samples':eligible,'correct_over_assessable':values['correct']/eligible if eligible else None}


def review_map(rows,expected):
    if len(rows)!=len({r['id'] for r in rows}) or {r['id'] for r in rows}!=set(expected):
        raise ValueError('Review coverage missing, duplicated or unexpected')
    for row in rows:
        if row['verdict'] not in {'correct','incorrect','uncertain'} or not row['reason'].strip():
            raise ValueError('Invalid review label or empty rationale')
    return {r['id']:r for r in rows}


def card(case,prediction,review):
    esc=html.escape
    visual=''
    if case['task']=='image':visual=f'<img loading="lazy" src="{esc(case["input_image"],quote=True)}" alt="{case["id"]} actual input image">'
    if case['task']=='video':visual=f'<img loading="lazy" src="inputs/{case["id"]}/input-8-frames.png" alt="{case["id"]} actual eight sampled video input frames">'
    return f'''<details id="{case['id']}" class="{review['verdict']}"><summary>{case['id']} · {esc(case['category'])} · {review['verdict']}</summary>{visual}
<h4>输入问题</h4><pre>{esc(case['prompt'])}</pre><h4>参考答案（数据集标注可能有误，非模型输出）</h4><pre>{esc(case['reference'])}</pre>
<h4>模型实际完整输出</h4><pre>{esc(prediction['completion'])}</pre><p>生成token（含EOS）：{len(prediction['completion_token_ids'])}；EOS停止：{prediction['stopped_at_eos']}</p>
<h4>单一AI复核</h4><p>{esc(review['reason'])}</p></details>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True,type=Path);args=parser.parse_args();p=args.folder.resolve()
    manifest=json.loads((p/'manifest.json').read_text());protocol=json.loads((p/'protocol.json').read_text())
    if manifest['status']!='completed' or sha(p/'predictions.jsonl')!=manifest['predictions_sha256'] or sha(p/'protocol.json')!=manifest['protocol_sha256']:
        raise ValueError('Completed, hash-bound inference required')
    cases=protocol['cases'];predictions=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
    if len(cases)!=300 or [r['id'] for r in cases]!=[r['id'] for r in predictions]:raise ValueError('All 300 original cases required in frozen order')
    with (p/'reviews.tsv').open() as f:reviews=review_map(list(csv.DictReader(f,delimiter='\t')),[c['id'] for c in cases])
    counts={};assets=[];categories=defaultdict(list)
    for case,pred in zip(cases,predictions):
        if pred['prompt']!=case['prompt'] or pred['reference']!=case['reference']:raise ValueError('Prompt/reference drift')
        n=len(pred['completion_token_ids'])
        if not 1<=n<=128 or (not pred['stopped_at_eos'] and n!=128):raise ValueError('Generation budget inconsistent')
        categories[(case['task'],case['category'])].append(reviews[case['id']]['verdict'])
        relative=case.get('input_image') if case['task']=='image' else f"inputs/{case['id']}/input-8-frames.png" if case['task']=='video' else None
        if relative:
            path=(p/relative).resolve()
            if not path.is_relative_to(p) or not path.is_file():raise ValueError('Input asset missing or escaping folder')
            digest=sha(path)
            if case['task']=='image' and digest!=case['input_sha256']:raise ValueError('Input image hash mismatch')
            assets.append({'id':case['id'],'file':relative,'sha256':digest})
    for task in ['text','image','video']:
        rows=[r for r in predictions if r['task']==task]
        if len(rows)!=100:raise ValueError('Exactly 100 per task required')
        values=summary_counts(reviews[r['id']]['verdict'] for r in rows)
        values.update(normalized_reference_exact_matches=sum(normalize(r['completion'])==normalize(r['reference']) for r in rows),
                      eos_stops=sum(r['stopped_at_eos'] for r in rows),hit_128_token_cap=sum(not r['stopped_at_eos'] for r in rows))
        counts[task]=values
    summary={'eval_id':manifest['eval_id'],'training_run_id':manifest['run_id'],'reviewed_at_utc':datetime.now(UTC).isoformat(),
             'counts':counts,'by_category':[{'task':task,'category':cat,**summary_counts(vals)} for (task,cat),vals in sorted(categories.items())],
             'protocol_sha256':sha(p/'protocol.json'),'predictions_sha256':sha(p/'predictions.jsonl'),'reviews_sha256':sha(p/'reviews.tsv'),
             'inference_manifest_sha256':sha(p/'manifest.json'),'summarizer_sha256':sha(__file__),
             'judge':'Single project-assistant semantic review, not independent human or multi-judge validated accuracy',
             'limits':'Different task sources/difficulty; no across-model ranking. Text is assistant-authored diagnostics. Images from previously used validation, video from previously observed QIVD test. Uncertain inputs retained, not resampled. No new training or answer-dependent prompt retries.'}
    (p/'review-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (p/'gallery-assets.json').write_text(json.dumps(assets,indent=2)+'\n')
    rows=''.join(f"<tr><td>{task}</td><td>{v['samples']}</td><td>{v['correct']}</td><td>{v['incorrect']}</td><td>{v['uncertain']}</td><td>{v['correct_over_all']:.1%}</td><td>{v['correct_over_assessable']:.2%}</td></tr>" for task,v in counts.items())
    content=''.join(card(c,r,reviews[c['id']]) for c,r in zip(cases,predictions))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MiniMind · 300条实际输入输出</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;max-width:1100px;margin:35px auto;padding:0 24px;color:#182838;background:#f7f9fc}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #ccd5df;padding:9px;text-align:left}}details{{background:white;border:1px solid #ccd5df;border-left:6px solid #c15858;border-radius:6px;margin:14px 0;padding:14px}}details.correct{{border-left-color:#23866b}}details.uncertain{{border-left-color:#b48225}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:12px;font:16px/1.65 system-ui}}img{{max-width:100%;max-height:620px;object-fit:contain;display:block;margin:18px auto}}small{{overflow-wrap:anywhere}}</style>
<h1>MiniMind：文本／图片／视频，各100条实际测试</h1><p>每个卡片可展开查看输入、参考、模型完整原文和评分理由。成功、失败与无法判定全部保留。图片是实际输入；视频图是模型实际使用的8个采样帧，不是生成视频、也不是终端截图。</p>
<p><strong>评分限制：</strong>本页为单一AI语义复核，不是独立人工标注。通过率=正确/100；可判定正确率=正确/(100−无法判定)。无法判定不是模型答对，也不删除或补抽。文本含20道明确格式约束题，违背约束计错。复读和无依据补充即使核心答案可接受也保留，不能当作良好对话体验。</p>
<table><tr><th>任务</th><th>样本</th><th>正确</th><th>错误</th><th>无法判定</th><th>正确/100</th><th>排除无法判定后</th></tr>{rows}</table>
<p>100张图仅做文件字节去重，不保证语义去重。三项来源和难度不同，不作横向能力排名。图像和QIVD素材仅在本地与服务器保留，用于本项目研究复核，不上传GitHub或重新公开分发。数据来源与许可见仓库 reports/data-manifest.md。</p>
<small>Eval ID: {html.escape(manifest['eval_id'])}<br>Frozen protocol SHA256: {summary['protocol_sha256']}<br>Actual predictions SHA256: {summary['predictions_sha256']}<br>Review labels SHA256: {summary['reviews_sha256']}</small>{content}</html>'''
    (p/'results.html').write_text(page,encoding='utf-8')
    examples={'T004','T005','I026','I073','V002','V091'}
    preview=''.join(card(c,r,reviews[c['id']]).replace('<details ','<details open ',1) for c,r in zip(cases,predictions) if c['id'] in examples)
    example_page=(page[:page.index('<details id=')]+preview+'</html>').replace('loading="lazy"','loading="eager"')
    (p/'examples.html').write_text(example_page,encoding='utf-8')
    for c,r in zip(cases,predictions):
        if c['id'] in examples:
            single=page[:page.index('<h1>')]+f"<h1>{c['id']} · 实际输入与完整输出</h1>"+card(c,r,reviews[c['id']]).replace('<details ','<details open ',1).replace('loading="lazy"','loading="eager"')+'</html>'
            (p/f"example-{c['id']}.html").write_text(single,encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
