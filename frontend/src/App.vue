<script setup>
import AppHeader from './components/AppHeader.vue'
</script>

<template>
  <div class="app-shell">
    <AppHeader />
    <main class="app-main">
      <RouterView />
    </main>

    <footer class="app-footer">
      <div class="app-footer__inner">
        <p>毕业设计系统：基于 SMILES 与分子指纹融合的分子毒性预测平台</p>
        <p>技术栈：PyTorch · RDKit · FastAPI · Vue3</p>
      </div>
    </footer>
  </div>
</template>