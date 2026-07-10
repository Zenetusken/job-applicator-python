#set page(margin: 0.75in)
#set text(font: "Libertinus Serif", size: 10pt)

{% if resume.source_sections %}
#align(center)[
{% for line in resume.source_preamble %}
{% if line.is_blank %}
  #v(2pt)
{% elif loop.first %}
  #text(size: 20pt, weight: "bold")[{{ line.text | typst_escape }}]
{% else %}
  #linebreak()
  #text(size: 9pt)[{{ line.text | typst_escape }}]
{% endif %}
{% endfor %}
]
#v(8pt)

{% for section in resume.source_sections %}
== {{ section.heading | typst_escape }}
#v(4pt)
{% for line in section.lines %}
{% if line.is_blank %}
#v(3pt)
{% elif line.is_bullet %}
- {{ line.text | typst_escape }}
{% else %}
{{ line.text | typst_escape }}
#linebreak()
{% endif %}
{% endfor %}
#v(6pt)
{% endfor %}
{% else %}
#align(center)[#text(size: 20pt, weight: "bold")[{{ resume.name | typst_escape }}]]
{% if resume.summary %}
#v(8pt)
== Summary
{{ resume.summary | typst_escape }}
{% endif %}
{% for exp in resume.experience %}
#v(6pt)
*{{ exp.title | typst_escape }}* — {{ exp.company | typst_escape }}
{% for bullet in exp.bullets %}
- {{ bullet | typst_escape }}
{% endfor %}
{% endfor %}
{% endif %}
