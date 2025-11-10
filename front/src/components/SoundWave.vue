<template>
  <div class="sound-wave-container">
    <div class="bar"></div> <div class="bar"></div> <div class="bar"></div>
    <div class="bar"></div> <div class="bar"></div> <div class="bar"></div>
    <div class="bar"></div> <div class="bar"></div> <div class="bar"></div>
    <div class="bar"></div>
  </div>
</template>
<style scoped>
.sound-wave-container { display: flex; justify-content: center; align-items: center; height: 50px; }
.bar { width: 6px; height: 5px; margin: 0 3px; background-color: #4f46e5; border-radius: 5px; animation: wave-animation 1.2s ease-in-out infinite; }
.bar:nth-child(2) { animation-delay: 0.1s; } .bar:nth-child(3) { animation-delay: 0.2s; }
.bar:nth-child(4) { animation-delay: 0.3s; } .bar:nth-child(5) { animation-delay: 0.4s; }
.bar:nth-child(6) { animation-delay: 0.5s; } .bar:nth-child(7) { animation-delay: 0.6s; }
.bar:nth-child(8) { animation-delay: 0.7s; } .bar:nth-child(9) { animation-delay: 0.8s; }
.bar:nth-child(10) { animation-delay: 0.9s; }
@keyframes wave-animation { 0% { transform: scaleY(0.5); } 50% { transform: scaleY(1.5); } 100% { transform: scaleY(0.5); } }
</style>