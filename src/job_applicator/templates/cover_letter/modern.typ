#set page(margin: 1in)
#set text(font: "Libertinus Serif", size: 11pt)

#align(right)[
  #text(weight: "bold")[{{ cover_letter.signature | typst_escape }}]
  {% if resume.email %}\
  {{ resume.email | typst_escape }}{% endif %}
  {% if resume.phone %}\
  {{ resume.phone | typst_escape }}{% endif %}
  {% if resume.location %}\
  {{ resume.location | typst_escape }}{% endif %}
  \
  {{ cover_letter.date | typst_escape }}
]

#v(16pt)

{{ cover_letter.recipient_company | typst_escape }}
{% if cover_letter.recipient_address %}\
{{ cover_letter.recipient_address | typst_escape }}{% endif %}

#v(16pt)

{{ cover_letter.greeting | typst_escape }}

#v(10pt)

{% for paragraph in cover_letter.paragraphs %}
{{ paragraph | typst_escape }}

#v(8pt)
{% endfor %}

{{ cover_letter.closing | typst_escape }},

#v(24pt)

{{ cover_letter.signature | typst_escape }}
