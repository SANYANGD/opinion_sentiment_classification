package com.example.demo.controller;

import com.alibaba.fastjson.JSON;
import com.example.demo.bean.Comment;
import com.example.demo.dao.commentDao;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.List;

@RestController
public class commentController {
    @Autowired
    commentDao dao;

    @RequestMapping("/getComment")
    public String getComment(String blog_id){
        String status = "fail";
        List<Comment> commentList = dao.getComment(blog_id);

        if(commentList != null)
            status = "success";

        HashMap<String, Object> map = new HashMap<>();
        map.put("status", status);
        map.put("object", commentList);

        return JSON.toJSONString(map);
    }
}
