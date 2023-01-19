<template>
  <div>
    <!--主卡片-->
    <el-card class="main-card">
      <el-card class="blog-card" shadow="hover">
        <div ref="myChart" style="height: 500px"></div>
      </el-card>
      <el-card v-for="(blog, i) in blogList" :key="i" class="blog-card" shadow="hover">
        <!--头部-->
        <div style="text-align: right">
          <el-avatar size="large" :src="blog.user_avatar"></el-avatar>
          <div class="user_name">{{blog.user_name}}</div>
        </div>
        <!--头部-->
        <el-divider content-position="left">微博原文</el-divider>
        <!--内容-->
        <div style="margin-top: 40px; line-height: 2">
          {{blog.text}}
        </div>
        <!--内容-->
        <el-divider></el-divider>
        <el-row>
          <el-col :span="24" style="text-align: center">
            <div class="grid-content bg-purple">
              <el-link
                  :underline="false"
                  style="margin-top: 12px; font-size: 20px; font-weight: bold"
                  icon="el-icon-s-comment" @click="getComment(i)">
                查看评论
              </el-link>
            </div>
          </el-col>
        </el-row>
      </el-card>
    </el-card>
    <!--弹出框-->
    <el-dialog title="评论一览" :visible.sync="visible">
      <el-card
          v-for="(comment, i) in commentList"
          :key="i"
          style="margin-top: 10px; box-shadow: 2px 2px 2px rgba(0, 8, 10, 0.15) !important;">
        <!--头部-->
        <div style="text-align: right">
          <el-avatar size="large" :src="comment.user_avatar"></el-avatar>
          <div class="user_name">{{comment.user_name}}</div>
        </div>
        <!--头部-->
        <el-divider content-position="left">用户评论</el-divider>
        <!--内容-->
        <div style="margin-top: 40px; line-height: 2; font-size: 20px;">
          {{comment.text}}
        </div>
        <!--内容-->
      </el-card>
    </el-dialog>
  </div>
</template>

<script>

export default {
  name: 'Home',
  mounted() {
    this.getData();
  },
  data() {
    return {
      blogList: [],
      commentList: [],
      visible: false,
    }
  },
  methods: {
    async getComment(index) {
      this.commentList = [];
      const {data: res} = await this.$http.get("getComment?blog_id=" + this.blogList[index].blog_id);
      if (res.status === "success") {
        this.commentList = res.object;
        this.visible = true;
      } else {
        this.$message.error("获取失败");
      }
    },

    async getData() {
      // 获取微博数据
      const {data: res} = await this.$http.get("getBlog");
      if (res.status === "success") {
        this.$message.success("获取成功");
        this.blogList = res.object;
      } else {
        this.$message.error("获取失败");
      }

      // 为绘制图表作数据准备
      let dataList = [];
      let indexList = [];
      for (let i = 0; i < res.object.length; i++) {
        dataList.push(res.object[i].analyse_result);
        indexList.push(i + 1);
      }

      // 绘制图表
      let myChart = this.$echarts.init(this.$refs.myChart, 'dark');
      // 指定图表的配置项和数据
      let option = {
        title: {
          text: "各微博积极评论占比一览",
        },
        xAxis: {
          type: 'category',
          data: indexList
        },
        yAxis: {
          type: 'value',
          name: "单位(%)"
        },
        series: [
          {
            data: dataList,
            type: 'bar',
            showBackground: true,
            backgroundStyle: {
              color: 'rgba(180, 180, 180, 0.2)'
            },
            label: {
              show: true,
              position: 'top',
              color: 'white',
              fontSize: 16
            },
          }
        ]
      };

      // 使用刚指定的配置项和数据显示图表。
      myChart.setOption(option)
    }
  }
}
</script>

<style>
.main-card {
  width: 1000px;
  margin-left: 28%;

  overflow-y: scroll;
  overflow-x: hidden;

  box-shadow: 2px 2px 2px rgba(0, 8, 10, 0.15) !important;
}
.blog-card {
  height: 100%;
  margin-top: 10px;
}
.user_name {
  font-weight: bold;
  font-size: 20px;

  height: 35px;
}
.el-col {
  border-radius: 4px;
}
.bg-purple {
  background: #d3dce6;
}
.grid-content {
  border-radius: 4px;
  min-height: 50px;
}
</style>
