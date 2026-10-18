"""Render the retained final F report."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def interval(values, scale=100, digits=4):
    return '[' + ', '.join(f'{scale*x:+.{digits}f}' for x in values) + ']'


def final_report(report):
    metrics = report['metrics']
    lines = [
        '# 最终 F：教师与融合统一评测', '',
        '同一固定 F：512 查询 / 10459 商品对。分类并列报告 **accuracy 和 Macro-F1**，分别评价总体正确率与类别平均表现；nDCG@10 评价排序。模型按单学生、教师、融合的方案顺序列出，不合成总分或完整排名。', '',
        '数据源：[analysis.json](analysis.json)。当前展示规则：[evaluation_policy.json](../../configs/evaluation_policy.json)。', '',
        '## 完整指标', '',
        '| 模型 | accuracy | Macro-F1 | nDCG@10 |',
        '|---|---:|---:|---:|',
    ]
    for name in metrics:
        m = metrics[name]
        lines.append(f"| {name} | {m['accuracy']:.6f} | {m['macro_f1']:.6f} | {m['ndcg10']:.6f} |")
    lines += ['',
        'S_C50 和 TA2_C50 分别在各自低置信度的 5230 条上等权融合，其余使用学生。S_Full 和 TA2_Full 在全量样本上等权融合，作为相同教师请求量的补充对照。推理样本选择和融合权重均不使用 F 标签。', '',
        '## Accuracy 配对比较', '',
        '以下差值和区间单位均为**百分点**，采用 2000 次完整查询配对 Bootstrap，seed=20260930。纠正指右侧系统判错、左侧判对；破坏反之。每项为单项区间。', '',
        '| 比较 | Δaccuracy（百分点） | 查询配对 CI95（百分点） | 纠正 / 破坏 | 净正确 |',
        '|---|---:|---|---:|---:|',
    ]
    for label, c in report['comparisons'].items():
        corrected, broken = c['corrected'], c['broken']
        lines.append(f"| {label} | {100*c['point']['accuracy']:+.4f} | {interval(c['CI95']['accuracy'])} | {corrected} / {broken} | {corrected-broken:+d} |")
    lines += ['',
        'TA2−G2 的 accuracy 差为 +0.3346 个百分点，区间 [+0.0661,+0.6189]，净正确 +35，支持同预算、同筛选样本下教师概率监督的额外准确率收益。'
        'TA2−T 为 +2.6867 个百分点，区间为正，学生推理仅用 0.6B 模型。S_C50 相对 S、T 分别提高 +2.2373、+3.9966 个百分点，两项区间均为正。', '',
        'TA2−A2 的 accuracy 差为 +0.1817 个百分点，区间 [−0.3507,+0.6935]，该项准确率优势未确认。TA2_C50−TA2 的区间也跨零，两种半量系统分别报告。', '',
        '## Macro-F1 配对比较', '',
        '| 比较 | ΔMacro-F1（百分点） | 查询配对 CI95（百分点） |',
        '|---|---:|---|',
    ]
    for label, c in report['comparisons'].items():
        lines.append(f"| {label} | {100*c['point']['macro_f1']:+.4f} | {interval(c['CI95']['macro_f1'])} |")
    lines += ['',
        'TA2−A2 的 Macro-F1 区间为正，而 accuracy 区间跨零；TA2−G2 和 TA2−T 的 accuracy 区间为正，而 Macro-F1 区间跨零。S_C50 相对 S、T 的两项分类差值区间均为正。并列报告各项证据，不将一个指标的优势扩展到另一个。', '',
        '## nDCG@10 配对比较（排序）', '',
        '此表使用原始 nDCG 数值差，未乘 100。', '',
        '| 比较 | ΔnDCG@10 | 查询配对 CI95 |',
        '|---|---:|---|',
    ]
    for label, c in report['comparisons'].items():
        lines.append(f"| {label} | {c['point']['ndcg10']:+.6f} | {interval(c['CI95']['ndcg10'],scale=1,digits=6)} |")
    lines += ['',
        '教师 nDCG@10=0.803842，高于两种半量融合。分类准确率优势不代表排序全面更强。', '',
        '## 预算、异常与评测范围', '',
        '教师模型：deepseek-flash；fingerprint：aeb56401ca74e127821c4f9126dcb669。完整教师缓存 10459 条，各条请求一次；实际采集成本按保守历史费率核算上界 3.434274 元。模拟半量部署每批 5230 次主请求。', '',
        'G2 的样本筛选依赖教师正确性；A2 是同训练预算、无教师监督对照。S 已做 ESCI SFT。所有数据在同一固定 F 上比较，学生训练为单 seed，区间条件于固定模型和教师响应。', '',
        '教师不可用 4 条。T 在完整 F 上将弃答计分类错误，保留全部样本；排序缺失分数为 0。融合按原规则回退学生，两种半量系统各有 2 条回退。', '',
        '即使假设 4 条不可用样本全部答对，教师 accuracy 上界也仅为 0.532651，Macro-F1 上界为 0.501733，仍低于半量融合。这是敏感性诊断，不是实测成绩。',
    ]
    return '\n'.join(lines) + '\n'



def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = read('results/final/analysis.json')
    path = ROOT / 'results/final/RESULT.md'
    path.write_text(final_report(report), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
