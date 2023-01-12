<template>
  <div id="Index">
    <AppBar></AppBar>
    <MainPage></MainPage>
    <MiRollsInt></MiRollsInt>
    <PageFooter></PageFooter>
  </div>
</template>
<script>

import AppBar from "@/components/AppBar.vue";
import MainPage from "@/components/MainPage.vue";
import MiRollsInt from "@/components/MiRollsInt.vue";
import PageFooter from "@/components/PageFooter.vue";
export default {
  name:"IndexPage",
  components: {PageFooter, MiRollsInt, MainPage, AppBar}
}
</script>
<style>
#Index{
  height: 100%;
}
</style>