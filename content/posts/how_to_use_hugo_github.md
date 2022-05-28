---
title: "Ubuntu下使用Hugo在GitHub Pages上搭建免费个人网站"
date: 2022-05-25
tags: ["blog"]
draft: false
---
### １．安装配置Hugo，也可参考[官方网站](https://gohugo.io/getting-started/quick-start/)
* 安装hugo
```
apt install hugo
```
* 创建网站
后面的-f yml是配置文件的格式，也可以不要，这意思是选择yml格式的配置文件，有些主题推荐某些格式
```
hugo new site github_blog　-f yml
```  
* 选择一个主题
可以去官方的[主题网站](https://themes.gohugo.io/)，我选择的是PaperMod,直接在github上找到，可以参考教程
```
cd github_blog
git init
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```
主题添加之后找到主题的配置文件覆盖本地的配置文件config.yml．
* 写一篇文章
```
hugo new posts/my-first-post.md
```
* 建立本地服务
```
hugo server -D
```
本地网址打开：http://localhost:1313/

如果觉得网页没有问题了便可以使用如下命令：

```
hugo
```

如此一来网页便生成在默认的public子目录中了。

### 2.发布并托管到github上
* 上传到Github之前，先在Github中添加一个空白repository，注意不要添加如README，.gitignore等文档 **注意：名称格式最好为：用户名.github.io**
![](/img/how_to_use_hugo_github/github.png)

```
cd github_blog
git add .
git commit -m "first commit"
git remote add origin https://github.com/xunathan/xunathan.github.io.git
git push -u origin main
```
至此所有源文档就都push到Github上了。然而此时Github对待这些源文档跟其他任何普通的repository中的代码并没有任何不同，并不会将public子目录中的网页托管在Github Pages上。
hugo官方文档上有两种方式让我们的Github Pages加载public中的网页：
我选择的是：**配置Hugo将网页生成在名为/docs的子目录中，然后直接push到main branch**，这样做的优点是本地整个hugo网站也可以在github托管维护，github　pages只加载/docs下的网页．
* 首先网页生成到/docs目录，修改配置文件(publishDir:docs)
![](/img/how_to_use_hugo_github/config.png)

自此,运行hugo命令后生成的网页文件将保存在/docs子目录下，要运行下hugo，然后commit修改并push到github.

* 修改github pages显示docs目录下的网页(setting->Pages),修改Source为下图
![](/img/how_to_use_hugo_github/github_pages.png)

等待片刻即可访问http://your_name.github.io看到之前用Hugo生成的网页了。

### 3.写文章并上传到网站
写文章的话比较简单，在content/posts增加markdown，然后运行hugo生成网页，接着提交到github上．

### 4.个人域名
可以在setting->Pages中找到custom domain设置你的个人域名．

### 5.markdown处理本地图片
本地图片放到static下img路径下
```
cd github_blog/static
mkdir img
```
makrdown中插入图片语法如下:
```
![](/img/how_to_use_hugo_github/config.png)
```