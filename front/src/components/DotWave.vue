<template>
  <div class="dot-wave">
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
  </div>
</template>

<script setup>
// 它总是处于动画状态
</script>

<style scoped>
.dot-wave {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px; /* 增加了点之间的间距 */
  height: 50px; /* 给容器一个高度，使其在布局中稳定 */
}

.dot {
  width: 8px; /* 增大了点的大小 */
  height: 8px;
  background-color: #a0a0c7; /* 默认颜色 */
  border-radius: 50%;
  animation: wave-animation 1.3s ease-in-out infinite; /* 调整了动画时间 */
}

/* 通过设置不同的 animation-delay 让动画看起来像波浪一样传播 */
.dot:nth-child(2) { animation-delay: 0.1s; }
.dot:nth-child(3) { animation-delay: 0.2s; }
.dot:nth-child(4) { animation-delay: 0.3s; }
.dot:nth-child(5) { animation-delay: 0.4s; }

@keyframes wave-animation {
  0%, 60%, 100% {
    transform: scaleY(1);
    background-color: #a0a0c7;
  }
  30% {
    transform: scaleY(2.5);
    background-color: #fff; /* 在动画顶点时变亮 */
  }
}
</style>