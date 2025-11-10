<script setup>
// 导入 lucide-vue-next 库中的图标组件
import { Power, Mic } from 'lucide-vue-next';

// 定义组件接收的 props
defineProps({
  // 'status' prop 用于决定显示哪个图标
  status: {
    type: String,
    required: true, // 这个 prop 是必需的
  },
});

// 定义组件可以向父组件发出的事件
const emit = defineEmits(['toggle-session']);
</script>

<template>
  <footer class="footer">
    <!-- 点击这个 div 时，向父组件发出 'toggle-session' 事件 -->
    <div class="mic-button-wrapper" @click="emit('toggle-session')">
      <!-- 根据 status 的值动态地显示 Power 或 Mic 图标 -->
      <Power v-if="status !== 'idle'" :size="28" color="#fff" />
      <Mic v-else :size="28" color="#fff" />
    </div>
  </footer>
</template>

<style scoped>
/* 这些样式定义了底部按钮的外观和交互效果 */
.footer { display: flex; justify-content: center; align-items: center; padding: 20px; padding-bottom: 40px; flex-shrink: 0; width: 100%; box-sizing: border-box; }
.mic-button-wrapper { width: 64px; height: 64px; background: rgba(255, 255, 255, 0.1); border-radius: 50%; display: flex; justify-content: center; align-items: center; cursor: pointer; transition: all 0.2s; }
.mic-button-wrapper:active { transform: scale(0.95); background: rgba(255, 255, 255, 0.2); }
.mic-button-wrapper:hover { background: rgba(255, 255, 255, 0.15); }
</style>