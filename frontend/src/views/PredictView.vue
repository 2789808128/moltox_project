<script setup>
import { ref, computed } from 'vue'

const smiles = ref('CCO')
const modelType = ref('fusion')
const displayMode = ref('single') // single | compare
const loading = ref(false)
const errorMsg = ref('')
const result = ref(null)
const compareResults = ref([])

const modelOptions = [
  { label: 'Transformer', value: 'transformer' },
  { label: 'Morgan + Logistic Regression', value: 'morgan_logreg' },
  { label: 'Morgan + Random Forest', value: 'morgan_rf' },
  { label: 'Fusion', value: 'fusion' },
]

const compareModelOrder = [
  { label: 'Transformer', value: 'transformer' },
  { label: 'Morgan + Logistic Regression', value: 'morgan_logreg' },
  { label: 'Morgan + Random Forest', value: 'morgan_rf' },
  { label: 'Fusion', value: 'fusion' },
]

const modelDescriptions = {
  transformer: '基于字符级 SMILES 序列建模的 Transformer 多任务毒性预测模型。',
  morgan_logreg: '基于 Morgan Fingerprint 的 Logistic Regression 传统机器学习基线模型。',
  morgan_rf: '基于 Morgan Fingerprint 的 Random Forest 传统机器学习基线模型。',
  fusion: '融合 SMILES Transformer 表征与 Morgan Fingerprint 表征的改进模型。',
}

const exampleSmiles = [
  { name: 'Ethanol', smiles: 'CCO' },
  { name: 'Benzene', smiles: 'c1ccccc1' },
  { name: 'Positive A', smiles: 'C=CC(=O)OCCOc1ccccc1' },
  { name: 'Positive B', smiles: 'O=C(Nc1ccccc1)c1ccccc1O' },
  { name: 'Positive C', smiles: 'Cc1ccc(N)cc1O' },
]

const taskRows = computed(() => {
  if (!result.value) return []

  const probs = result.value.task_probs || {}
  const preds = result.value.task_preds || {}

  return Object.keys(probs).map((task) => ({
    task,
    prob: probs[task],
    pred: preds[task],
  }))
})

const positiveCount = computed(() => {
  if (!result.value) return 0
  return Object.values(result.value.task_preds || {}).filter((v) => v === 1).length
})

const maxRiskTask = computed(() => {
  if (!result.value) return null

  const probs = result.value.task_probs || {}
  let bestTask = null
  let bestProb = -1

  for (const [task, prob] of Object.entries(probs)) {
    if (prob !== null && prob > bestProb) {
      bestTask = task
      bestProb = prob
    }
  }

  if (bestTask === null) return null
  return {
    task: bestTask,
    prob: bestProb,
  }
})

const riskLevel = computed(() => {
  if (!result.value || !maxRiskTask.value) return 'N/A'
  const p = maxRiskTask.value.prob
  if (p >= 0.5) return '高'
  if (p >= 0.2) return '中'
  return '低'
})

const currentModelLabel = computed(() => {
  const item = modelOptions.find((x) => x.value === modelType.value)
  return item ? item.label : modelType.value
})

const currentModelDescription = computed(() => {
  return modelDescriptions[modelType.value] || ''
})

const compareSummaryRows = computed(() => {
  return compareResults.value.map((item) => {
    const probs = item.task_probs || {}
    const preds = item.task_preds || {}

    let bestTask = null
    let bestProb = -1
    for (const [task, prob] of Object.entries(probs)) {
      if (prob !== null && prob > bestProb) {
        bestTask = task
        bestProb = prob
      }
    }

    const positiveTasks = Object.values(preds).filter((v) => v === 1).length

    return {
      model_name: item.model_name,
      label: compareModelOrder.find((x) => x.value === item.model_name)?.label || item.model_name,
      positiveCount: positiveTasks,
      bestTask,
      bestProb,
    }
  })
})

function fillExample(item) {
  smiles.value = item.smiles
}

function clearInput() {
  smiles.value = ''
  result.value = null
  compareResults.value = []
  errorMsg.value = ''
}

async function fetchSinglePrediction(selectedModel, inputSmiles) {
  const response = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_type: selectedModel,
      smiles: inputSmiles,
    }),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.detail || '预测失败。')
  }

  return data
}

async function handlePredict() {
  errorMsg.value = ''
  result.value = null
  compareResults.value = []

  if (!smiles.value.trim()) {
    errorMsg.value = '请输入 SMILES 字符串。'
    return
  }

  loading.value = true

  try {
    const inputSmiles = smiles.value.trim()

    if (displayMode.value === 'single') {
      const data = await fetchSinglePrediction(modelType.value, inputSmiles)
      result.value = data
    } else {
      const promises = compareModelOrder.map((item) =>
        fetchSinglePrediction(item.value, inputSmiles)
      )
      const allResults = await Promise.all(promises)
      compareResults.value = allResults
    }
  } catch (err) {
    errorMsg.value = err.message || '请求失败。'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="page-section">
    <div class="page-hero">
      <span class="page-kicker">Prediction Center</span>
      <h2>模型预测与结果展示</h2>
      <p>
        选择不同模型，对输入分子的 SMILES 字符串进行 Tox21 多任务毒性预测，并展示各毒性终点的概率与分类结果。
      </p>
    </div>

    <div class="content-card predict-form-card">
      <div class="predict-grid">
        <div class="left-panel">
          <div class="form-group">
            <label for="smiles">SMILES 输入</label>
            <textarea
              id="smiles"
              v-model="smiles"
              rows="5"
              placeholder="请输入 SMILES，例如：CCO"
            />
          </div>

          <div class="example-block">
            <div class="example-title">示例分子</div>
            <div class="example-buttons">
              <button
                v-for="item in exampleSmiles"
                :key="item.name"
                class="example-btn"
                @click="fillExample(item)"
              >
                {{ item.name }}
              </button>
            </div>
          </div>
        </div>

        <div class="right-panel">
          <div class="form-group">
            <label>结果展示模式</label>
            <div class="mode-switch">
              <button
                class="mode-btn"
                :class="{ active: displayMode === 'single' }"
                @click="displayMode = 'single'"
              >
                单模型预测
              </button>
              <button
                class="mode-btn"
                :class="{ active: displayMode === 'compare' }"
                @click="displayMode = 'compare'"
              >
                四模型对比
              </button>
            </div>
          </div>

          <div class="form-group" v-if="displayMode === 'single'">
            <label for="modelType">模型选择</label>
            <select id="modelType" v-model="modelType">
              <option
                v-for="item in modelOptions"
                :key="item.value"
                :value="item.value"
              >
                {{ item.label }}
              </option>
            </select>
          </div>

          <div class="model-desc-card">
            <div class="model-desc-title">
              {{ displayMode === 'single' ? '当前模型说明' : '模型差异说明' }}
            </div>
            <p v-if="displayMode === 'single'">
              {{ currentModelDescription }}
            </p>
            <p v-else>
              同一分子在不同模型下可能得到不同预测结果，这是正常现象。Transformer 更关注 SMILES
              序列模式，Morgan 指纹模型更关注局部结构片段，而 Fusion 模型会综合两类信息后重新学习决策边界。
              因此，多模型对比能够更直观地展示不同分子表示方式与不同决策机制的差异。
            </p>
          </div>

          <div class="button-row">
            <button class="predict-btn" @click="handlePredict" :disabled="loading">
              {{ loading ? '预测中...' : '开始预测' }}
            </button>
            <button class="clear-btn" @click="clearInput" :disabled="loading">
              清空输入
            </button>
          </div>

          <p v-if="errorMsg" class="error-text">{{ errorMsg }}</p>
        </div>
      </div>
    </div>

    <div v-if="displayMode === 'single' && result" class="content-card result-card">
      <div class="result-topbar">
        <div>
          <span class="result-model-badge">当前模型：{{ currentModelLabel }}</span>
          <h3>预测结果</h3>
          <p><strong>SMILES：</strong>{{ result.smiles }}</p>
        </div>
      </div>

      <div class="summary-grid">
        <div class="summary-card">
          <div class="summary-label">阳性任务数</div>
          <div class="summary-value">{{ positiveCount }} / 12</div>
        </div>

        <div class="summary-card" v-if="maxRiskTask">
          <div class="summary-label">最高风险终点</div>
          <div class="summary-value summary-small">
            {{ maxRiskTask.task }}
          </div>
          <div class="summary-subtext">
            概率：{{ Number(maxRiskTask.prob).toFixed(4) }}
          </div>
        </div>

        <div class="summary-card">
          <div class="summary-label">风险等级</div>
          <div class="summary-value">{{ riskLevel }}</div>
        </div>
      </div>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>任务名称</th>
              <th>概率值</th>
              <th>预测结果</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in taskRows" :key="row.task">
              <td>{{ row.task }}</td>
              <td>
                <div class="prob-cell">
                  <span class="prob-text">{{ Number(row.prob).toFixed(4) }}</span>
                  <div class="prob-bar">
                    <div
                      class="prob-fill"
                      :style="{ width: `${Math.max(0, Math.min(100, row.prob * 100))}%` }"
                    />
                  </div>
                </div>
              </td>
              <td>
                <span
                  class="pred-badge"
                  :class="row.pred === 1 ? 'positive' : 'negative'"
                >
                  {{ row.pred }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div v-if="displayMode === 'compare' && compareSummaryRows.length > 0" class="content-card compare-card">
      <div class="result-topbar">
        <div>
          <span class="result-model-badge">四模型对比模式</span>
          <h3>多模型对比结果</h3>
          <p><strong>SMILES：</strong>{{ smiles }}</p>
        </div>
      </div>

      <div class="compare-grid">
        <div class="compare-model-card" v-for="item in compareSummaryRows" :key="item.model_name">
          <div class="compare-model-title">{{ item.label }}</div>
          <div class="compare-metric">
            <span class="compare-label">阳性任务数</span>
            <span class="compare-value">{{ item.positiveCount }} / 12</span>
          </div>
          <div class="compare-metric">
            <span class="compare-label">最高风险终点</span>
            <span class="compare-value compare-small">{{ item.bestTask || 'N/A' }}</span>
          </div>
          <div class="compare-metric">
            <span class="compare-label">最大概率</span>
            <span class="compare-value">
              {{ item.bestProb !== null && item.bestProb !== undefined ? Number(item.bestProb).toFixed(4) : 'N/A' }}
            </span>
          </div>
        </div>
      </div>

      <div class="compare-note">
        <p>
          提示：同一分子在不同模型下可能出现不同预测结果，这反映了不同分子表示方式与不同决策机制的差异。
          该功能用于突出本系统的多模型对比分析能力。
        </p>
      </div>
    </div>

    <div class="content-card chart-card">
      <h3>模型实验结果展示</h3>
      <p class="chart-desc">
        下方展示四模型平均 AUC 对比图，以及训练过程曲线图，用于展示模型性能与优化趋势。
      </p>

      <div class="chart-grid">
        <div class="chart-item">
          <h4>四模型平均 AUC 对比</h4>
          <img src="/model_comparison_mean_auc.png" alt="Model Comparison" />
        </div>

        <div class="chart-item">
          <h4>Transformer 训练曲线</h4>
          <img src="/transformer_curves.png" alt="Transformer Training Curves" />
        </div>

        <div class="chart-item">
          <h4>Fusion 训练曲线</h4>
          <img src="/fusion_curves.png" alt="Fusion Training Curves" />
        </div>

        <div class="chart-item chart-wide">
          <h4>各任务 ROC-AUC 对比</h4>
          <img src="/task_auc_comparison.png" alt="Task AUC Comparison" />
        </div>
      </div>
    </div>
  </section>
</template>