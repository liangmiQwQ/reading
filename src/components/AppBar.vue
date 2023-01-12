<template>
    <div id="bar">
        <span id="title">米卷</span>
        <span id="secTitle">wj.lmfans.cn</span>
        <titleBtn innerHtml="首页" @click="goHome" />
        <titleBtn innerHtml="问卷结果查询" @click="resSearch" />
        <titleBtn innerHtml="制作问卷" @click="makeQtn" />
        <!-- <LoginOrRegister loginStates="true"></LoginOrRegister> -->
        <LoginOrRegister loginStates userHeadImg="./favicon.png"></LoginOrRegister>
    </div>
</template>
<script>
    import titleBtn from "./titleBtn.vue";
    import LoginOrRegister from "./LoginOrRegister.vue"
    export default {
        name: 'AppBar',
        methods: {
            goHome() {
                alert("这个功能还没写完捏")
            },
            resSearch() {
                alert("这个功能还没写完捏")
            },
            makeQtn() {
                alert("这个功能还没写完捏")
            }
        },
        components: {
            titleBtn,
            LoginOrRegister
        }
    }

</script>
<style>
    #bar {
        background-color: rgb(21, 127, 248);
        width: 100%;
        height: 33px;
        padding-top: 20px;
        padding-bottom: 20px;
        box-shadow: 0px 0px 12px rgba(0, 0, 0, 0.581);
        color: rgb(255, 242, 202);
        position: fixed;
        top: 0;
        margin-top: 0;
    }

    #title {
        margin-left: 100px;
        margin-right: 5%;
        line-height: 28px;
        font-size: 28px;
        font-weight: 400;
    }

    #secTitle {
        position: absolute;
        top: 48px;
        left: 100px;
        font-size: 5px;
        margin-left: 3px;
    }
</style>