#set page(margin: 1in)
#set text(font: "Libertinus Serif", size: 11pt)

#text(size: 18pt, weight: "bold")[{{ resume.name | typst_escape }}]
{% if resume.title %}
#v(2pt)
#text(size: 11pt, style: "italic")[{{ resume.title | typst_escape }}]
{% endif %}
#v(6pt)
{% set contacts = [] %}
{% if resume.email %}{% set _ = contacts.append(resume.email | typst_escape) %}{% endif %}
{% if resume.phone %}{% set _ = contacts.append(resume.phone | typst_escape) %}{% endif %}
{% if resume.location %}{% set _ = contacts.append(resume.location | typst_escape) %}{% endif %}
{% if resume.linkedin_url %}{% set _ = contacts.append(resume.linkedin_url | typst_escape) %}{% endif %}
{% if resume.portfolio_url %}{% set _ = contacts.append(resume.portfolio_url | typst_escape) %}{% endif %}
{{ contacts | join(" · ") }}

{% if resume.summary %}
#v(14pt)
{{ resume.summary | typst_escape }}
{% endif %}

#v(18pt)
== Experience
#v(6pt)
{% for exp in resume.experience %}
#text(weight: "bold")[{{ exp.title | typst_escape }}] #h(1fr) {{ exp.start_date | typst_escape }}{% if exp.end_date %} – {{ exp.end_date | typst_escape }}{% endif %} \
{{ exp.company | typst_escape }}{% if exp.location %}, {{ exp.location | typst_escape }}{% endif %}
{% for bullet in exp.bullets %}
- {{ bullet | typst_escape }}
{% endfor %}
#v(8pt)
{% endfor %}

{% if resume.education %}
#v(8pt)
== Education
#v(6pt)
{% for edu in resume.education %}
#text(weight: "bold")[{{ edu.degree | typst_escape }}] #h(1fr) {% if edu.start_date %}{{ edu.start_date | typst_escape }}{% if edu.end_date %} – {{ edu.end_date | typst_escape }}{% endif %}{% endif %} \
{{ edu.institution | typst_escape }}{% if edu.location %}, {{ edu.location | typst_escape }}{% endif %}
{% endfor %}
{% endif %}

{% if resume.skills %}
#v(8pt)
== Skills
#v(6pt)
{% for group in resume.skills %}
{% if group.category %}#text(weight: "bold")[{{ group.category | typst_escape }}:] {% endif %}{{ group.skills | join(", ") | typst_escape }}
{% endfor %}
{% endif %}

{% if resume.certifications %}
#v(8pt)
== Certifications
#v(6pt)
{% for cert in resume.certifications %}
- {{ cert | typst_escape }}
{% endfor %}
{% endif %}

{% if resume.languages %}
#v(8pt)
== Languages
#v(6pt)
{{ resume.languages | join(", ") | typst_escape }}
{% endif %}

{% if resume.projects %}
#v(8pt)
== Projects
#v(6pt)
{% for project in resume.projects %}
#text(weight: "bold")[{{ project.name | typst_escape }}]{% if project.url %} — {{ project.url | typst_escape }}{% endif %}{% if project.description %}: {{ project.description | typst_escape }}{% endif %}
{% endfor %}
{% endif %}
