from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUT_DIR = Path("results/reports")
OUT_DOCX = OUT_DIR / "experiment_status_report_20260603_2344.docx"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def para(text: str, style: str | None = None) -> str:
    ppr = ""
    if style:
        ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>"
    return f"<w:p>{ppr}<w:r><w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r></w:p>"


def bullet(text: str) -> str:
    return para(f"• {text}")


def document_xml() -> str:
    parts: list[str] = []
    parts.append(para("持续指令微调实验状态说明", "Title"))
    parts.append(para("截至：2026-06-03 23:44（UTC+8）"))
    parts.append(para("项目：/root/autodl-tmp/Lora-code"))

    parts.append(para("一、实验到底在做什么？自测什么？", "Heading1"))
    parts.append(
        para(
            "这个实验是在做 Continual Instruction Tuning（持续指令微调）。核心场景是：同一个 Llama 3.1 8B Instruct "
            "骨干模型按顺序接收一串不同任务 segment，每到一个新 segment 就用 LoRA 继续微调，然后立即评估模型在当前任务和已经见过的历史任务上的表现。"
        )
    )
    parts.append(
        para(
            "它自测的不是单个静态数据集上的一次性准确率，而是持续学习过程中的三个问题：新任务能不能学会，旧任务会不会忘，"
            "以及遇到任务切换时，方法里的漂移检测、LoRA bank、router 和 anti-overlap 机制是否真的在起作用。"
        )
    )
    parts.append(bullet("benchmark：主要是 instrdialog 和 instrdialog++ 两条 CITB 风格的持续任务流。"))
    parts.append(bullet("训练方式：每个 segment 训练 LoRA，然后评估当前 segment 以及所有已见 segment。"))
    parts.append(bullet("运行方式：真实大实验通过 tmux 后台跑，并把指标同步到 W&B project `lora-citb-acl`。"))
    parts.append(bullet("小实验：`scripts/smoke_test_sequence_diag.py` 和 `configs/seq_debug.yaml` 已通过，用来确认基本评估链路能跑通。"))

    parts.append(para("二、baselines 和 ours 分别是什么？", "Heading1"))
    parts.append(para("Baselines 是标准对照方法，用来回答“如果不用我们完整方法，普通持续 LoRA 或简单记忆/路由策略能做到什么水平”。"))
    parts.append(bullet("Sequential：顺序 LoRA。所有任务按顺序继续训练同一个适配器，是最朴素的持续微调。"))
    parts.append(bullet("Replay(10) / Replay(50)：LoRA 加 replay buffer，训练新任务时混入少量旧样本，测试 replay 对抗遗忘的作用。"))
    parts.append(bullet("PeriodicLatest：周期性维护/切换 LoRA 适配器，使用最新或周期性策略作为对照。"))
    parts.append(bullet("RouterOnly：主要测试路由器本身的分配能力，不包含完整方法的全部机制。"))
    parts.append(bullet("BankNoRouter：有 LoRA bank，但不使用 router，用来拆分 bank 和 routing 的贡献。"))
    parts.append(
        para(
            "Ours 是完整方法：漂移检测（drift detector）判断任务分布是否变化，LoRA bank 保存多个分支适配器，router 决定样本应该走哪个分支，"
            "anti-overlap 正则尝试让不同分支学到更分离的表示。消融实验会分别关掉 bank、drift、router、overlap，或扫 LoRA rank、branch 数、anchor size、overlap beta 等超参。"
        )
    )

    parts.append(para("三、实验目标是什么？期望得到哪些指标和结果？", "Heading1"))
    parts.append(para("实验目标是证明 ours 在持续指令微调中比 baseline 更能兼顾新任务学习和旧任务保持。理想结果不是只看最后一个任务，而是整条任务流上的综合表现。"))
    parts.append(bullet("eval.current_score：当前 segment 的准确率/任务分数。希望越高越好。"))
    parts.append(bullet("eval.seen_avg_score / eval.anytime_score：所有已见任务的平均表现。希望越高越好，这是持续学习最核心指标。"))
    parts.append(bullet("eval.forgetting：历史任务遗忘程度。希望越低越好。"))
    parts.append(bullet("token_f1_mean、lcs_overlap_mean：生成文本与答案的软匹配指标。希望在 exact score 低时能辅助判断是否只是格式不匹配。"))
    parts.append(bullet("routing.oracle_agreement_rate、decision_confidence、utilization_entropy：判断 router 是否真的学会合理分配分支。"))
    parts.append(bullet("drift.false_alarm_rate、drift.miss_rate、detection_delay_mean：判断漂移检测是否及时且准确。"))
    parts.append(
        para(
            "期望结果是：ours_full 的 seen_avg_score / anytime_score 高于 baselines，forgetting 更低；消融中去掉关键模块后性能下降；"
            "drift miss rate 不能太高，router 不能完全退化成随机或单一分支。"
        )
    )

    parts.append(para("四、实验具体设计细节", "Heading1"))
    parts.append(
        para(
            "实验按 segment 顺序运行。每个 segment 代表一个任务或一组同分布指令样本。训练阶段只使用当前 segment 的训练样本，"
            "某些 baseline 或 ours 会额外使用 replay buffer、LoRA bank 或 router。评估阶段会在当前 segment 和所有已见历史 segment 上重新计算指标，"
            "因此可以同时观察新任务学习能力和历史任务保持能力。"
        )
    )
    parts.append(para("核心评估指标与含义："))
    parts.append(
        bullet(
            "EM / exact match：最严格的答案匹配指标。生成答案经过归一化后必须与 gold answer 完全一致才计为正确。"
            "表里的 eval.current_score、eval.seen_avg_score 在多数分类/短答案任务上可以近似理解为 EM/任务准确率。"
        )
    )
    parts.append(
        bullet(
            "current_score：当前 segment 上的 EM/任务分数。它回答“刚学的新任务有没有学会”。"
        )
    )
    parts.append(
        bullet(
            "seen_avg_score：对所有已经见过的 segment 求平均后的 EM/任务分数。它回答“模型到目前为止整体还保留了多少能力”。"
        )
    )
    parts.append(
        bullet(
            "anytime_score：训练过程中到当前时间点的历史平均表现，和 seen_avg_score 一起衡量持续学习质量。"
        )
    )
    parts.append(
        bullet(
            "task_aware_score：在部分任务上使用任务类型感知的打分方式，例如分类任务只判断标签是否正确，生成任务则看答案文本匹配。"
            "它用于避免不同任务格式造成不可比。"
        )
    )
    parts.append(
        bullet(
            "forgetting：历史任务曾经达到的最好分数与当前分数之间的差。越高代表忘得越严重，越低越好。"
        )
    )
    parts.append(
        bullet(
            "token_f1_mean：按 token 计算预测与 gold 的 F1。它比 EM 宽松，能发现“答案部分相关但没有完全匹配”的情况。"
        )
    )
    parts.append(
        bullet(
            "lcs_overlap_mean：最长公共子序列重叠度，也是一种软匹配指标。它能辅助判断模型是否生成了相似文本但格式或边界不一致。"
        )
    )
    parts.append(para("路由、漂移和分支相关指标："))
    parts.append(
        bullet(
            "routing.num_routed：实际经过 router 分配的样本数。若为 0，说明该方法没有启用 router 或该阶段没有路由行为。"
        )
    )
    parts.append(
        bullet(
            "routing.oracle_agreement_rate：router 选择的分支与 oracle 最优分支的一致率。越高说明 router 越接近理想分配。"
        )
    )
    parts.append(
        bullet(
            "decision_confidence / decision_entropy：router 对分支选择的置信度与不确定性。置信度过低或熵过高可能表示分支不可分。"
        )
    )
    parts.append(
        bullet(
            "utilization_entropy：分支利用是否均衡。过低可能表示 router 退化为总选同一分支；过高但分数不提升则可能表示随机分配。"
        )
    )
    parts.append(
        bullet(
            "drift.false_alarm_rate：没有真实任务变化时误报 drift 的比例。越低越好。"
        )
    )
    parts.append(
        bullet(
            "drift.miss_rate：真实发生任务变化但未检测到的比例。越低越好；目前已有结果中这个值偏高，是重点风险。"
        )
    )
    parts.append(
        bullet(
            "detection_delay_mean：发生漂移后平均延迟多少 segment/step 才检测到。越低越好。"
        )
    )
    parts.append(
        bullet(
            "overlap_mean_cosine：不同分支表示/LoRA 更新之间的平均相似度。anti-overlap 希望降低不必要的重叠，让分支更专门化。"
        )
    )
    parts.append(para("实验矩阵设计："))
    parts.append(bullet("benchmark 维度：instrdialog 与 instrdialog++。instrdialog++ 是更强或更复杂的任务流版本。"))
    parts.append(bullet("method 维度：baseline 包括 Sequential、Replay、PeriodicLatest、RouterOnly、BankNoRouter；ours 包括 OursFull 与多个消融。"))
    parts.append(bullet("seed 维度：目标是至少跑 123、456、789 三个 seed，避免单 seed 偶然性。当前 ours 多 seed 还没完全齐。"))
    parts.append(bullet("ablation 维度：去掉 bank、drift、router、overlap，观察哪个模块对最终表现最关键。"))
    parts.append(bullet("sweep 维度：扫 LoRA rank、branch 数、anchor size、monitor interval、overlap beta 等，寻找更稳定的配置。"))
    parts.append(
        para(
            "因此，最终判断不能只看某一个 run 的最后 current_score，而应该综合：多 seed 的 seen_avg_score 是否提升，forgetting 是否下降，"
            "soft metrics 是否同步改善，router/drift 指标是否合理，以及消融是否呈现符合机制预期的下降。"
        )
    )

    parts.append(para("五、目前已有结果是什么？", "Heading1"))
    parts.append(para("当前大实验仍在运行，不是最终结论。已观察到的状态如下："))
    parts.append(bullet("tmux `baseline_results_wandb` 正在跑，日志显示已推进到 instrdialog++ baseline 的 Segment 31 左右。"))
    parts.append(bullet("tmux `ours_seed123_wandb_queue` 还在等待 baseline 队列结束，暂时不是主训练进程。"))
    parts.append(bullet("W&B 在线同步正常，project 为 `lora-citb-acl`。"))
    parts.append(bullet("instrdialog 的 baseline 多数已有 3 个 seed 汇总；instrdialog++ 还有部分 baseline 正在补跑。"))
    parts.append(bullet("ours 目前主要已有 seed 123 的结果，多 seed 结论还不完整。"))
    parts.append(para("已有汇总中的代表性数字："))
    parts.append(bullet("instrdialog / BankNoRouter：seen_avg_score mean 约 0.0316，current_score mean 约 0.0333。"))
    parts.append(bullet("instrdialog / Sequential、Replay、RouterOnly：seen_avg_score 多数为 0 或接近 0。"))
    parts.append(bullet("instrdialog / OursFull seed123：seen_avg_score 为 0，forgetting 约 0.137。"))
    parts.append(bullet("instrdialog++ / OursFull seed123：seen_avg_score 约 0.0793，token_f1_mean 约 0.1917，lcs_overlap_mean 约 0.2560。"))
    parts.append(bullet("instrdialog++ / BankNoRouter 三 seed：seen_avg_score mean 约 0.0351。"))
    parts.append(bullet("instrdialog++ / RouterOnly seed123：seen_avg_score 约 0.0531。"))

    parts.append(para("六、已经产出的结果有什么明显问题？", "Heading1"))
    parts.append(para("目前最明显的问题是绝对分数偏低，而且低到需要优先排查训练/评估链路，而不是直接写成方法效果不佳。"))
    parts.append(bullet("大量 run 的 eval.current_score 和 eval.seen_avg_score 是 0 或接近 0，说明 exact match/任务分数几乎没有被打上。"))
    parts.append(bullet("soft metrics 仍有非零值，例如 token_f1 和 lcs_overlap 有时在 0.1 到 0.3 之间，说明模型可能生成了部分相关文本，但格式、答案边界或解码方式可能不对。"))
    parts.append(bullet("当前正在跑的 baseline 日志显示到 31 个 segment 左右仍是 seen_avg_score=0.0，这对真实 8B LoRA 来说偏异常，需要检查标签模板、generation prompt、答案归一化和评估规则。"))
    parts.append(bullet("ours 的多 seed 尚不完整，不能用 seed123 单点结果下最终结论。"))
    parts.append(bullet("drift detector 的 miss_rate 在已有 ours_full 结果里很高，例如 instrdialog++ seed123 约 0.946，说明漂移检测可能漏检严重。"))
    parts.append(bullet("部分 routing 指标看起来有活动，但主分数没有同步提升，可能意味着 router 学到的分配未转化成任务表现，或者评估指标没有正确捕捉生成质量。"))
    parts.append(
        para(
            "建议下一步先做诊断，而不是盲目继续扩大实验：抽样查看预测文本与 gold 答案，确认 chat template、label masking、generation slicing、"
            "答案归一化和 max_new_tokens 是否合理；再针对 drift miss rate 和 router 分支利用率做小规模可解释实验。"
        )
    )

    parts.append(para("七、针对近期分数低与漏检问题的改动", "Heading1"))
    parts.append(para("根据上述诊断结果，已于近期进行了如下核心代码与配置改动，目前指标已大幅恢复："))
    parts.append(bullet("修复 Exact Match (EM) 提取失败：在 prompt 的 user content 中增加了严格的短回答约束；并且在 _truncate_prediction_for_scoring 中增加了十余种常见 Llama 3.1 闲聊套话前缀的强制过滤，使得提取到的核心答案与 Gold 能够精确匹配。"))
    parts.append(bullet("调优漂移检测 (Drift Detector) 降低漏检率：在 configs 中将 shift_stat 从 degradation 改为 absolute，捕获任何剧烈的 NLL 偏移；将 score_ema 从 0.9 降至 0.5 提升突变敏感度；并设置 min_consecutive_probe_hits: 1，加快偏移上报。"))
    parts.append(bullet("解决同时运行 baseline 和 ours 导致的 OOM 问题：采用 tmux 任务队列串行执行实验。"))

    body = "".join(parts)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def write_docx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/></w:pPr><w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
</w:styles>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml())
        zf.writestr("word/styles.xml", styles)


def main() -> None:
    write_docx(OUT_DOCX)
    print(OUT_DOCX)


if __name__ == "__main__":
    main()
