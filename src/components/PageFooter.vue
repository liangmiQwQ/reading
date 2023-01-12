<template>
  <div class="footer">
    <h1>米卷</h1>
    <p class="footerButton">一个免费且好用的国内问卷软件</p>
    <p class="footerButton" style="cursor: pointer;" @click="goIcp">ICP备：</p>
  </div>
</template>
<script>
  export default{
    name:"PageFooter",
    methods:{
      goIcp(){
        window.location.href=""
      }
    }
  }
</script>
<style>
.footer{
  padding: 50px 0 50px 0;
  margin-top: 50px;
  background: rgb(23,27,33);
  text-align: center;
  color: white;
}
.footerButton{
  font-size: 17px;
}
</style>