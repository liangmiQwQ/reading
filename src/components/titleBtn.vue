<template>
    <button id="btn" @click="$emit('click')">{{ innerHtml }}</button>
</template>
<script>
    export default {
        name: 'titleBtn',
        props: {
            innerHtml: String
        },

    }
</script>
<style>
    #btn {
        height: 33px;
        border: none;
        font-size: 18px;
        padding: 0 13px 0 13px;
        background: none;
        color: white;
        transition: 0.1s;
        font-weight: bold;
        margin-left: 50px;
        cursor: pointer;
    }

    #btn:hover {
        color: rgb(247, 242, 98);
    }
</style>