<script setup>
import { RouterLink, useRoute } from 'vue-router'

const route = useRoute()

const navItems = [
  { label: '首页总览', to: '/' },
  { label: '方法介绍', to: '/knowledge' },
  { label: '模型预测', to: '/predict' },
  { label: '系统设计', to: '/system-design' },
]
</script>

<template>
  <header class="app-header">
    <div class="app-header__inner">
      <div class="app-logo">
        <span class="app-logo__badge">MT</span>
        <div class="app-logo__text">
          <h1>MolTox System</h1>
          <p>Drug Toxicity Prediction Platform</p>
        </div>
      </div>

      <nav class="app-nav">
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="app-nav__link"
          :class="{ 'is-active': route.path === item.to }"
        >
          {{ item.label }}
        </RouterLink>
      </nav>
    </div>
  </header>
</template>