<script setup>
import { computed, ref } from 'vue'

const fragmentOptions = [
  { key: 'aromatic_ring', label: '芳香环', bits: [2, 5, 9, 14], desc: '芳香环通常对应较稳定的环状局部结构特征。' },
  { key: 'hydroxyl', label: '羟基', bits: [1, 7, 12], desc: '羟基会引入含氧官能团相关的局部特征。' },
  { key: 'carboxyl', label: '羧基', bits: [4, 8, 15, 18], desc: '羧基常与极性和酸性相关的结构特征有关。' },
  { key: 'amine', label: '胺基', bits: [3, 10, 16], desc: '胺基会激活含氮局部环境的结构位点。' },
  { key: 'halogen', label: '卤素', bits: [6, 11, 17], desc: '卤素取代基可能改变分子的疏水性和反应活性。' },
]

const selectedFragments = ref([])

function toggleFragment(key) {
  if (selectedFragments.value.includes(key)) {
    selectedFragments.value = selectedFragments.value.filter((item) => item !== key)
  } else {
    selectedFragments.value.push(key)
  }
}

const activeBits = computed(() => {
  const bitSet = new Set()

  fragmentOptions.forEach((item) => {
    if (selectedFragments.value.includes(item.key)) {
      item.bits.forEach((b) => bitSet.add(b))
    }
  })

  return Array.from(bitSet).sort((a, b) => a - b)
})

const activeDescriptions = computed(() => {
  return fragmentOptions.filter((item) => selectedFragments.value.includes(item.key))
})

const bitVector = computed(() => {
  const bits = Array.from({ length: 20 }, (_, i) => ({
    index: i,
    active: activeBits.value.includes(i),
  }))
  return bits
})

const summaryText = computed(() => {
  if (selectedFragments.value.length === 0) {
    return '请选择一个或多个局部结构片段，观察它们如何点亮“指纹位”。'
  }

  return `当前已选择 ${selectedFragments.value.length} 个局部结构片段，共激活 ${activeBits.value.length} 个示意指纹位。`
})
</script>

<template>
  <div class="fingerprint-game">
    <div class="fingerprint-game__header">
      <h3>分子指纹互动小游戏</h3>
      <p>
        点击下方局部结构片段，观察右侧“示意指纹位”如何被点亮，
        从而理解“结构片段会映射成固定长度向量特征”的基本思想。
      </p>
    </div>

    <div class="fingerprint-game__layout">
      <div class="fingerprint-game__left">
        <div class="fingerprint-fragment-list">
          <button
            v-for="item in fragmentOptions"
            :key="item.key"
            class="fingerprint-fragment-btn"
            :class="{ active: selectedFragments.includes(item.key) }"
            @click="toggleFragment(item.key)"
          >
            {{ item.label }}
          </button>
        </div>

        <div class="fingerprint-summary-box">
          <div class="fingerprint-summary-title">当前说明</div>
          <p>{{ summaryText }}</p>
        </div>

        <div v-if="activeDescriptions.length > 0" class="fingerprint-desc-list">
          <div
            v-for="item in activeDescriptions"
            :key="item.key"
            class="fingerprint-desc-item"
          >
            <h4>{{ item.label }}</h4>
            <p>{{ item.desc }}</p>
          </div>
        </div>
      </div>

      <div class="fingerprint-game__right">
        <div class="fingerprint-vector-card">
          <div class="fingerprint-vector-title">示意 Fingerprint Bit Vector</div>
          <div class="fingerprint-bit-grid">
            <div
              v-for="bit in bitVector"
              :key="bit.index"
              class="fingerprint-bit"
              :class="{ active: bit.active }"
            >
              <span class="bit-index">{{ bit.index }}</span>
              <span class="bit-value">{{ bit.active ? 1 : 0 }}</span>
            </div>
          </div>
        </div>

        <div class="fingerprint-note">
          <p>
            说明：这里展示的是教学用途的“简化指纹位向量示意”，
            用于帮助理解“局部结构 → 位向量特征”的映射思想，
            并非严格还原真实 Morgan 指纹每一位的精确化学语义。
          </p>
        </div>
      </div>
    </div>
  </div>
</template>