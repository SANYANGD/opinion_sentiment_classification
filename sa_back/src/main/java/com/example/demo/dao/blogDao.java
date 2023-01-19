package com.example.demo.dao;

import com.example.demo.bean.Blog;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface blogDao {
    List<Blog> getBlog();
}
