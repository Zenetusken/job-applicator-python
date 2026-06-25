#set page(margin: 0.75in)
#set text(font: "Libertinus Serif", size: 10pt)

#align(center)[
  #text(size: 20pt, weight: "bold")[{{ resume.name | typst_escape }}]
  {% if resume.title %}
  #linebreak()
  #text(size: 12pt, style: "italic")[{{ resume.title | typst_escape }}]
  {% endif %}
  #linebreak()
  #text(size: 9pt)[
    {% set contacts = [] %}
    {% if resume.email %}{% set _ = contacts.append(resume.email | typst_escape) %}{% endif %}
    {% if resume.phone %}{% set _ = contacts.append(resume.phone | typst_escape) %}{% endif %}
    {% if resume.location %}{% set _ = contacts.append(resume.location | typst_escape) %}{% endif %}
    {% if resume.linkedin_url %}{% set _ = contacts.append(resume.linkedin_url | typst_escape) %}{% endif %}
    {% if resume.portfolio_url %}{% set _ = contacts.append(resume.portfolio_url | typst_escape) %}{% endif %}
    {{ contacts | join(" · ") | typst_escape }}
  ]
]
#v(8pt)

{% if resume.summary %}
== Summary
{{ resume.summary | typst_escape }}
#v(6pt)
{% endif %}

== Experience
#v(4pt)
{% for exp in resume.experience %}
*{{ exp.title | typst_escape }}* — {{ exp.company | typst_escape }}{% if exp.location %}, {{ exp.location | typst_escape }}{% endif %} #h(1fr) {{ exp.start_date | typst_escape }}{% if exp.end_date %} – {{ exp.end_date | typst_escape }}{% endif %}
{% for bullet in exp.bullets %}
- {{ bullet | typst_escape }}
{% endfor %}
#v(4pt)
{% endfor %}

{% if resume.education %}
== Education
#v(4pt)
{% for edu in resume.education %}
*{{ edu.degree | typst_escape }}* — {{ edu.institution | typst_escape }}{% if edu.location %}, {{ edu.location | typst_escape }}{% endif %} #h(1fr) {% if edu.start_date %}{{ edu.start_date | typst_escape }}{% if edu.end_date %} – {{ edu.end_date | typst_escape }}{% endif %}{% endif %}
{% endfor %}
#v(6pt)
{% endif %}

{% if resume.skills %}
== Skills
#v(4pt)
{% for group in resume.skills %}
{% if group.category %}*{{ group.category | typst_escape }}:* {% endif %}{{ group.skills | join(", ") | typst_escape }}
{% endfor %}
#v(6pt)
{% endif %}

{% if resume.certifications %}
== Certifications
#v(4pt)
{% for cert in resume.certifications %}
- {{ cert | typst_escape }}
{% endfor %}
#v(6pt)
{% endif %}

{% if resume.languages %}
== Languages
#v(4pt)
{{ resume.languages | join(", ") | typst_escape }}
#v(6pt)
{% endif %}

{% if resume.projects %}
== Projects
#v(4pt)
{% for project in resume.projects %}
*{{ project.name | typst_escape }}*{% if project.url %} — {{ project.url | typst_escape }}{% endif %}{% if project.description %}: {{ project.description | typst_escape }}{% endif %}
{% endfor %}
{% endif %}
