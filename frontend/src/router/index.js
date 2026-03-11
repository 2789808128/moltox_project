import { createRouter, createWebHistory } from 'vue-router'

import HomeView from '../views/HomeView.vue'
import KnowledgeView from '../views/KnowledgeView.vue'
import PredictView from '../views/PredictView.vue'
import SystemDesignView from '../views/SystemDesignView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
  },
  {
    path: '/knowledge',
    name: 'knowledge',
    component: KnowledgeView,
  },
  {
    path: '/predict',
    name: 'predict',
    component: PredictView,
  },
  {
    path: '/system-design',
    name: 'system-design',
    component: SystemDesignView,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router