<template>
  <section class="page-section">
    <div class="page-hero">
      <span class="page-kicker">System Design</span>
      <h2>系统设计与软件工程过程展示</h2>
      <p>
        本页从软件工程视角对系统进行说明，包括需求分析、功能模块划分、系统架构设计、
        关键业务流程以及后续可扩展方向，用于支撑毕业设计系统实现与答辩展示。
      </p>
    </div>

    <div class="content-card">
      <h3>需求分析</h3>
      <div class="system-two-col">
        <div class="design-box">
          <h4>功能需求</h4>
          <ul>
            <li>支持输入 SMILES 字符串并进行毒性预测</li>
            <li>支持 Transformer、Morgan+LogReg、Morgan+RF、Fusion 四种模型切换</li>
            <li>支持展示 12 个毒性终点的预测概率与分类结果</li>
            <li>支持展示模型训练曲线、模型对比图与方法介绍内容</li>
            <li>支持面向使用者的知识说明与交互式教学模块</li>
          </ul>
        </div>

        <div class="design-box">
          <h4>非功能需求</h4>
          <ul>
            <li>界面友好，适合演示与答辩展示</li>
            <li>系统结构清晰，前后端职责明确</li>
            <li>模型调用统一化，便于后续扩展新模型</li>
            <li>结果返回规范化，便于前端可视化展示</li>
            <li>整体系统具备可维护性与可扩展性</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>系统功能模块划分</h3>
      <div class="module-grid">
        <div class="module-card">
          <h4>数据处理模块</h4>
          <p>完成原始数据清洗、合法性检查、标签统计与训练/验证/测试集划分。</p>
        </div>
        <div class="module-card">
          <h4>模型训练模块</h4>
          <p>负责 Transformer、传统机器学习基线与 Fusion 模型的训练与最优模型保存。</p>
        </div>
        <div class="module-card">
          <h4>统一推理模块</h4>
          <p>为不同模型提供统一 predictor 接口，支持路由分发与标准化结果输出。</p>
        </div>
        <div class="module-card">
          <h4>后端服务模块</h4>
          <p>通过 FastAPI 封装统一预测接口，接收前端请求并返回预测结果。</p>
        </div>
        <div class="module-card">
          <h4>前端展示模块</h4>
          <p>基于 Vue 实现多页面系统，展示项目概览、方法介绍、预测结果与系统设计内容。</p>
        </div>
        <div class="module-card">
          <h4>交互教学模块</h4>
          <p>通过分子指纹小游戏增强用户理解，提升系统沉浸感与演示效果。</p>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>系统总体架构</h3>
      <div class="arch-flow-grid">
        <div class="arch-flow-card">
          <div class="arch-flow-title">前端层</div>
          <p>Vue 多页面界面：总览页、方法介绍页、预测页、系统设计页</p>
        </div>
        <div class="arch-arrow">↓</div>
        <div class="arch-flow-card">
          <div class="arch-flow-title">服务层</div>
          <p>FastAPI 接口、请求响应模式、模型路由分发</p>
        </div>
        <div class="arch-arrow">↓</div>
        <div class="arch-flow-card">
          <div class="arch-flow-title">推理层</div>
          <p>Transformer / Morgan LogReg / Morgan RF / Fusion predictor</p>
        </div>
        <div class="arch-arrow">↓</div>
        <div class="arch-flow-card">
          <div class="arch-flow-title">模型与数据层</div>
          <p>模型 checkpoint、训练数据、tokenizer、Morgan fingerprint 表征</p>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>用例说明</h3>
      <div class="usecase-grid">
        <div class="usecase-card">
          <h4>用例 1：普通用户进行毒性预测</h4>
          <p>
            用户在预测页输入 SMILES，选择模型，点击预测按钮，
            系统调用后端接口并返回 12 个毒性终点的预测概率与分类结果。
          </p>
        </div>
        <div class="usecase-card">
          <h4>用例 2：答辩教师查看方法说明</h4>
          <p>
            教师通过方法介绍页查看 SMILES、分子指纹与毒性预测任务说明，
            并通过小游戏理解分子指纹与局部结构的关系。
          </p>
        </div>
        <div class="usecase-card">
          <h4>用例 3：评审查看系统设计过程</h4>
          <p>
            评审人员通过系统设计页查看系统需求、功能模块、架构设计与软件工程实现逻辑。
          </p>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>关键业务流程</h3>
      <div class="flow-step-grid">
        <div class="flow-step-card">
          <span class="flow-step-index">1</span>
          <h4>输入分子表示</h4>
          <p>用户输入 SMILES 字符串，系统接收并校验输入。</p>
        </div>
        <div class="flow-step-card">
          <span class="flow-step-index">2</span>
          <h4>模型选择</h4>
          <p>用户选择所需模型，前端将模型类型与分子字符串打包请求。</p>
        </div>
        <div class="flow-step-card">
          <span class="flow-step-index">3</span>
          <h4>后端推理</h4>
          <p>后端根据模型类型路由到对应 predictor，生成预测结果。</p>
        </div>
        <div class="flow-step-card">
          <span class="flow-step-index">4</span>
          <h4>结果展示</h4>
          <p>前端接收结果并展示毒性概率、分类结果、风险摘要与实验图表。</p>
        </div>
      </div>
    </div>

    <div class="content-card">
      <h3>可扩展方向</h3>
      <div class="system-two-col">
        <div class="design-box">
          <h4>模型扩展</h4>
          <ul>
            <li>引入预训练化学语言模型</li>
            <li>引入图神经网络或图 Transformer</li>
            <li>设计更复杂的融合策略</li>
          </ul>
        </div>

        <div class="design-box">
          <h4>系统扩展</h4>
          <ul>
            <li>增加用户登录与实验记录保存</li>
            <li>增加分子名称查询与结构可视化</li>
            <li>增加批量预测与结果导出功能</li>
          </ul>
        </div>
      </div>
    </div>
  </section>
</template>