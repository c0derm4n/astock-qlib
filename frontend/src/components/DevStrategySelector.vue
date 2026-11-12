<template>
  <div class="dev-strategy-switch" role="group" aria-label="选择正式策略">
    <button v-for="strategy in strategies" :key="strategy.id" type="button" :class="{ active: modelValue === strategy.id }" :style="{ '--strategy-color': strategy.color }" :aria-pressed="modelValue === strategy.id" @click="$emit('update:modelValue', strategy.id)"><i></i>{{ strategy.name }}</button>
  </div>
</template>
<script setup>
defineProps({ modelValue: String, strategies: { type: Array, default: () => [] } })
defineEmits(['update:modelValue'])
</script>
<style scoped>
.dev-strategy-switch { display: flex; flex-wrap: wrap; gap: 8px; }
button { display: flex; gap: 9px; align-items: center; min-width: 150px; background: white; border: 1px solid #dfe5eb; border-radius: 5px; padding: 11px 14px; color: #6e7d8e; cursor: pointer; font-size: 12px; font-family: inherit; line-height: 1.45; text-align: left; }
i { width: 6px; height: 6px; background: var(--strategy-color); border-radius: 50%; }
button.active { color: var(--strategy-color); border-color: var(--strategy-color); box-shadow: inset 0 -2px var(--strategy-color); }
button:hover { background: #f7f9fb; } button:focus-visible { outline: 2px solid #598bcf; outline-offset: 3px; }
@media(max-width: 600px) { button { flex: 1 1 calc(50% - 8px); min-width: 0; padding: 11px; justify-content: flex-start; } }
@media(max-width: 390px) { button { flex-basis: 100%; } }
</style>
