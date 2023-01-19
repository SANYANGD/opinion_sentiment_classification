package com.example.demo.controller;

import com.alibaba.fastjson.JSON;
import com.example.demo.bean.Blog;
import com.example.demo.dao.blogDao;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.List;
import java.util.Objects;

@RestController
public class blogController {
    @Autowired
    blogDao dao;

    @RequestMapping("/test")
    public String test(){
        return "Test my blogController";
    }

    @RequestMapping("/getBlog")
    public String getBlog(){
        String status = "fail";
        List<Blog> blogList = dao.getBlog();

        if(blogList != null)
            status = "success";

        HashMap<String, Object> map = new HashMap<>();
        map.put("status", status);
        map.put("object", blogList);

        return JSON.toJSONString(map);
    }
}
