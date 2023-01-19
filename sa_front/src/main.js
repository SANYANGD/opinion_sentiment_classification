import Vue from 'vue'
import App from './App.vue'
import router from './router'
import './assets/css/global.css'

import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css'
Vue.use(ElementUI);

import axios from "axios";
Vue.prototype.$http = axios
axios.defaults.baseURL = "http://localhost:9000"

import * as echarts from 'echarts';
Vue.prototype.$echarts = echarts

Vue.config.productionTip = false

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')
