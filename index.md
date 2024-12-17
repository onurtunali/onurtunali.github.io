---
layout: default
---

<h2 class="posts-item-note" aria-label="Recent Posts">Recent Posts</h2>
{%- for post in site.posts limit: site.number_of_posts -%}
<article class="post-item">
  <span class="post-item-date">{{ post.date | date: "%b %d, %Y" }}</span>
  <h4 class="post-item-title">
    <a href="{{ post.url }}">{{ post.title | escape }}</a>
  </h4>
</article>

<article class="post-item">
  <span class="post-item-date"></span>
  <span class="post-item-date">
  <a><p>{{ post.excerpt}}</p> </a>
  </span>
</article>

{%- endfor -%}
