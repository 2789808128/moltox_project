<script setup>
import FingerprintGame from '../components/FingerprintGame.vue'
</script>

<template>
  <section class="page-section">
    <div class="page-hero">
      <span class="page-kicker">Method Introduction</span>
      <h2>SMILES、分子指纹与毒性预测方法介绍</h2>
      <p>
        本页用于介绍系统所涉及的核心概念，包括 SMILES 表示、Morgan Fingerprint、
        毒性预测任务以及它们在模型中的作用。
      </p>
    </div>

    <div class="content-card">
      <h3>什么是 SMILES？</h3>
      <p class="section-text">
        SMILES（Simplified Molecular Input Line Entry System）是一种使用字符串表示分子结构的方法。
        它能够将分子中的原子、键、支链和环结构编码成线性文本，从而使分子结构可以被自然语言处理模型或序列模型直接读取。
      </p>
      <div class="knowledge-example-box">
        <div class="knowledge-example-item">
          <div class="knowledge-example-label">Ethanol</div>
          <div class="knowledge-example-code">CCO</div>
        </div>
        <div class="knowledge-example-item">
          <div class="knowledge-example-label">Benzene</div>
          <div class="knowledge-example-code">c1ccccc1</div>
        </div>
        <div class="knowledge-example-item">
          <div class="knowledge-example-label">Aspirin</div>
          <div class="knowledge-example-code">CC(=O)OC1=CC=CC=C1C(=O)O</div>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>什么是分子指纹？</h3>
      <p class="section-text">
        分子指纹是一种将分子结构映射为固定长度向量的方法。它通过统计或编码分子的局部子结构特征，
        形成一个可以直接输入到传统机器学习模型中的数值表示。
      </p>
      <p class="section-text">
        在本系统中，我们主要采用 Morgan Fingerprint。它通过对分子中每个原子及其邻域结构进行迭代编码，
        将局部结构模式映射到固定长度的 bit 向量，从而用于逻辑回归、随机森林以及融合模型中的结构分支。
      </p>
    </div>

    <div class="content-card">
      <h3>毒性预测任务是如何完成的？</h3>
      <p class="section-text">
        本系统面向 Tox21 数据集中的 12 个毒性终点任务，每个任务都可以看作一个独立的二分类问题。
        模型输入分子的 SMILES 表示或指纹表示，输出每个毒性任务为阳性的概率，并据此给出预测结果。
      </p>
      <div class="tox-flow-grid">
        <div class="tox-flow-card">
          <h4>输入层</h4>
          <p>SMILES 序列 / Morgan Fingerprint</p>
        </div>
        <div class="tox-flow-card">
          <h4>模型层</h4>
          <p>Transformer / 传统机器学习 / Fusion</p>
        </div>
        <div class="tox-flow-card">
          <h4>输出层</h4>
          <p>12 个毒性终点的概率与分类结果</p>
        </div>
      </div>
    </div>

    <div class="content-card">
      <FingerprintGame />
    </div>
  </section>
</template>