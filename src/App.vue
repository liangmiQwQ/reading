<template>
  <div id="app">
    <IndexPage></IndexPage>
  </div>
</template>

<script>
  import IndexPage from './pages/IndexPage.vue'
  export default {
    name: 'App',
    components: {
      IndexPage
    }
  }
</script>

<style>
  #app {
    width: 100%;
    -webkit-font-smoothing: antialiased;
    -moz-font-smoothing: grayscale;
    height: 100%;
  }
</style>