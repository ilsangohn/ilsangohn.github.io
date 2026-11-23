---
title: Research
subtitle: "[연구 프로그램을 2~3문장으로. 어떤 큰 질문을 풀고 있는지.]"
---

{% assign me = site.author.name %}
{% assign preprints = site.data.publications | where: "status", "preprint" %}
{% assign published = site.data.publications | where: "status", "published" | sort: "year" | reverse %}

{% if preprints.size > 0 %}
## Preprints
<ol class="pubs">
{% for p in preprints %}{% include pub.html p=p me=me %}{% endfor %}
</ol>
{% endif %}

## Publications
<ol class="pubs">
{% for p in published %}{% include pub.html p=p me=me %}{% endfor %}
</ol>
