<template>
  <section class="page-section">
    <div class="page-hero">
      <span class="page-kicker">Project Overview</span>
      <h2>基于 SMILES 与分子指纹融合的分子毒性预测系统</h2>
      <p>
        本系统面向药物毒性预测任务，围绕 Tox21 数据集，构建了基于字符级 SMILES 的 Transformer 模型、
        基于 Morgan Fingerprint 的传统机器学习模型，以及融合序列表征与结构指纹的改进模型，
        并进一步完成了统一推理接口、后端服务与前端可视化展示。
      </p>
    </div>

    <div class="content-card overview-highlight-card">
      <div class="overview-highlight-grid">
        <div class="overview-highlight-item">
          <div class="overview-highlight-label">数据集</div>
          <div class="overview-highlight-value">Tox21</div>
        </div>
        <div class="overview-highlight-item">
          <div class="overview-highlight-label">任务数</div>
          <div class="overview-highlight-value">12</div>
        </div>
        <div class="overview-highlight-item">
          <div class="overview-highlight-label">最佳传统基线</div>
          <div class="overview-highlight-value">Morgan + RF</div>
        </div>
        <div class="overview-highlight-item">
          <div class="overview-highlight-label">融合模型测试 AUC</div>
          <div class="overview-highlight-value">0.7974</div>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>研究背景</h3>
      <p class="section-text">
        药物毒性预测是药物研发中的关键环节。传统湿实验方法通常具有周期长、成本高、效率低等问题，
        因此利用机器学习与深度学习方法，从分子结构表示中自动学习毒性相关模式，成为当前计算药物学中的重要研究方向。
      </p>
      <p class="section-text">
        本项目选取 Tox21 多任务毒性预测数据集，围绕 SMILES 序列表示、分子指纹表示以及两者融合策略开展建模研究，
        并最终实现了一个可交互、可展示、可扩展的分子毒性预测系统。
      </p>
    </div>

    <div class="content-card">
      <h3>项目目标</h3>
      <div class="overview-list-grid">
        <div class="mini-feature-card">
          <h4>构建基础实验闭环</h4>
          <p>完成数据清洗、训练验证测试划分、模型训练、测试评估与可视化分析。</p>
        </div>
        <div class="mini-feature-card">
          <h4>建立多模型对照体系</h4>
          <p>实现 Transformer、Morgan+LogReg、Morgan+RF 与 Fusion 四类模型进行对比。</p>
        </div>
        <div class="mini-feature-card">
          <h4>验证融合策略有效性</h4>
          <p>分析 SMILES 序列表征与 Morgan 指纹表征的互补性，提升毒性预测性能。</p>
        </div>
        <div class="mini-feature-card">
          <h4>形成系统化展示平台</h4>
          <p>完成统一推理接口、后端 API 与 Vue 前端系统，支持可视化演示与毕设答辩展示。</p>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>技术路线</h3>
      <div class="route-grid">
        <div class="route-step">
          <div class="route-step-index">01</div>
          <h4>数据处理</h4>
          <p>读取 Tox21 数据，完成 SMILES 清洗、标签统计、训练/验证/测试划分。</p>
        </div>
        <div class="route-step">
          <div class="route-step-index">02</div>
          <h4>基线模型构建</h4>
          <p>搭建字符级 SMILES Transformer，以及 Morgan 指纹传统机器学习基线。</p>
        </div>
        <div class="route-step">
          <div class="route-step-index">03</div>
          <h4>融合模型设计</h4>
          <p>将 Transformer 分支与 Morgan Fingerprint 分支融合，形成改进模型。</p>
        </div>
        <div class="route-step">
          <div class="route-step-index">04</div>
          <h4>系统封装与展示</h4>
          <p>统一推理接口，接入 FastAPI 后端与 Vue 前端，实现可交互展示平台。</p>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>系统设计路线</h3>
      <div class="system-arch-grid">
        <div class="arch-block">
          <h4>数据层</h4>
          <p>负责原始数据读取、清洗、划分，以及训练和推理所需的数据表示准备。</p>
        </div>
        <div class="arch-block">
          <h4>模型层</h4>
          <p>负责 Transformer、传统 ML 基线、Fusion 模型训练与推理逻辑实现。</p>
        </div>
        <div class="arch-block">
          <h4>服务层</h4>
          <p>通过统一推理接口与 FastAPI 路由分发机制，提供统一预测服务。</p>
        </div>
        <div class="arch-block">
          <h4>展示层</h4>
          <p>基于 Vue 构建多页面前端，对模型结果、曲线与系统设计进行可视化展示。</p>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>实验结论摘要</h3>
      <div class="overview-list-grid">
        <div class="mini-feature-card">
          <h4>Transformer</h4>
          <p>测试集平均 ROC-AUC 为 <strong>0.7554</strong>，证明字符级 SMILES 序列建模能够完成毒性预测任务。</p>
        </div>
        <div class="mini-feature-card">
          <h4>Morgan + Logistic Regression</h4>
          <p>测试集平均 ROC-AUC 为 <strong>0.7858</strong>，说明传统分子指纹方法具有很强的结构表达能力。</p>
        </div>
        <div class="mini-feature-card">
          <h4>Morgan + Random Forest</h4>
          <p>测试集平均 ROC-AUC 为 <strong>0.8131</strong>，是当前实验中的最强传统基线模型。</p>
        </div>
        <div class="mini-feature-card">
          <h4>Fusion</h4>
          <p>测试集平均 ROC-AUC 为 <strong>0.7974</strong>，明显优于原始 Transformer，并超过 LogReg，接近 RF。</p>
        </div>
      </div>
      <p class="section-text" style="margin-top: 14px;">
        实验结果表明，SMILES 序列表征与 Morgan 指纹表征具有互补性，融合策略能够显著提升模型性能，
        为后续进一步引入更强预训练模型或图结构表示提供了清晰方向。
      </p>
    </div>

    <div class="content-card chart-card">
      <h3>模型结果概览</h3>
      <p class="chart-desc">
        当前实验共完成 Transformer、Morgan+LogReg、Morgan+RF 与 Fusion 四类模型，
        下图展示其在 Tox21 测试集上的平均 ROC-AUC 对比结果。
      </p>

      <div class="chart-grid">
        <div class="chart-item chart-wide">
          <h4>四模型平均 AUC 对比</h4>
          <img src="/model_comparison_mean_auc.png" alt="Model Comparison" />
        </div>
      </div>
    </div>
  </section>
</template>