// src/components/AssistantStatus.vue

<script setup>
// 定义组件期望接收的 props
defineProps({
  // title 是必须的字符串类型
  title: {
    type: String,
    required: true,
  },
  // subtitle 是可选的字符串类型
  subtitle: {
    type: String,
    default: '',
  },
});
</script>

<template>
  <div class="status-text-wrapper">
    <h2 class="status-title">{{ title }}</h2>
    <!-- 只有当 subtitle 有内容时才渲染这个 <p> 标签 -->
    <p class="status-subtitle" v-if="subtitle">{{ subtitle }}</p>
  </div>
</template>

<style scoped>
.status-text-wrapper { text-align: center; z-index: 2; margin-top: 40px; }
.status-title { font-size: 20px; font-weight: 500; margin: 0; min-height: 28px; }
.status-subtitle { font-size: 14px; color: rgba(255, 255, 255, 0.5); margin-top: 8px; }
</style>