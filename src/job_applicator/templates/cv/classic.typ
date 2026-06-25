#set page(margin: 1in)
#set text(font: "Libertinus Serif", size: 10.5pt)

#align(center)[
  #text(size: 22pt, weight: "bold")[{{ resume.name | typst_escape }}]
  {% if resume.title %}
  #linebreak()
  #text(size: 12pt)[{{ resume.title | typst_escape }}]
  {% endif %}
  #linebreak()
  #text(size: 9.5pt)[
    {% set contacts = [] %}
    {% if resume.email %}{% set _ = contacts.append(resume.email | typst_escape) %}{% endif %}
    {% if resume.phone %}{% set _ = contacts.append(resume.phone | typst_escape) %}{% endif %}
    {% if resume.location %}{% set _ = contacts.append(resume.location | typst_escape) %}{% endif %}
    {% if resume.linkedin_url %}{% set _ = contacts.append(resume.linkedin_url | typst_escape) %}{% endif %}
    {% if resume.portfolio_url %}{% set _ = contacts.append(resume.portfolio_url | typst_escape) %}{% endif %}
    {{ contacts | join(" | ") }}
  ]
]
#v(12pt)

{% if resume.summary %}
#align(center)[#text(size: 9pt, weight: "bold", tracking: 1pt)[SUMMARY]]
#line(length: 100%)
#v(2pt)
{{ resume.summary | typst_escape }}
#v(10pt)
{% endif %}

#align(center)[#text(size: 9pt, weight: "bold", tracking: 1pt)[EXPERIENCE]]
#line(length: 100%)
#v(4pt)
{% for exp in resume.experience %}
#grid(
  columns: (1fr, auto),
  gutter: 8pt,
  [*{{ exp.title | typst_escape }}* — {{ exp.company | typst_escape }}{% if exp.location %}, {{ exp.location | typst_escape }}{% endif %}],
  [{{ exp.start_date | typst_escape }}{% if exp.end_date %} – {{ exp.end_date | typst_escape }}{% endif %}],
)
{% for bullet in exp.bullets %}
- {{ bullet | typst_escape }}
{% endfor %}
#v(4pt)
{% endfor %}

{% if resume.education %}
#v(6pt)
#align(center)[#text(size: 9pt, weight: "bold", tracking: 1pt)[EDUCATION]]
#line(length: 100%)
#v(4pt)
{% for edu in resume.education %}
#grid(
  columns: (1fr, auto),
  gutter: 8pt,
  [*{{ edu.degree | typst_escape }}* — {{ edu.institution | typst_escape }}{% if edu.location %}, {{ edu.location | typst_escape }}{% endif %}],
  [{% if edu.start_date %}{{ edu.start_date | typst_escape }}{% if edu.end_date %} – {{ edu.end_date | typst_escape }}{% endif %}{% endif %}],
)
{% endfor %}
#v(6pt)
{% endif %}

{% if resume.skills %}
#v(6pt)
#align(center)[#text(size: 9pt, weight: "bold", tracking: 1pt)[SKILLS]]
#line(length: 100%)
#v(4pt)
{% for group in resume.skills %}
{% if group.category %}*{{ group.category | typst_escape }}:* {% endif %}{{ group.skills | join(", ") | typst_escape }}
{% endfor %}
#v(6pt)
{% endif %}

{% if resume.certifications %}
#v(6pt)
#align(center)[#text(size: 9pt, weight: "bold", tracking: 1pt)[CERTIFICATIONS]]
#line(length: 100%)
#v(4pt)
{% for cert in resume.certifications %}
- {{ cert | typst_escape }}
{% endfor %}
#v(6pt)
{% endif %}

{% if resume.languages %}
#v(6pt)
#align(center)[#text(size: 9pt, weight: "bold", tracking: 1pt)[LANGUAGES]]
#line(length: 100%)
#v(4pt)
{{ resume.languages | join(", ") | typst_escape }}
#v(6pt)
{% endif %}

{% if resume.projects %}
#v(6pt)
#align(center)[#text(size: 9pt, weight: "bold", tracking: 1pt)[PROJECTS]]
#line(length: 100%)
#v(4pt)
{% for project in resume.projects %}
*{{ project.name | typst_escape }}*{% if project.url %} — {{ project.url | typst_escape }}{% endif %}{% if project.description %}: {{ project.description | typst_escape }}{% endif %}
{% endfor %}
{% endif %}
