package com.example.demo.dao;

import com.example.demo.bean.Comment;
import org.apache.ibatis.annotations.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface commentDao {
    List<Comment> getComment(@Param("blog_id") String blog_id);
}
