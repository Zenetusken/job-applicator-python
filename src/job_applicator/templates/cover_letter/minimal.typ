#set page(margin: 1.25in)
#set text(font: "Libertinus Serif", size: 11pt)

{{ cover_letter.signature | typst_escape }}{% if resume.email %} / {{ resume.email | typst_escape }}{% endif %}{% if resume.phone %} / {{ resume.phone | typst_escape }}{% endif %}{% if resume.location %} / {{ resume.location | typst_escape }}{% endif %}

#v(8pt)

{{ cover_letter.date | typst_escape }}

#v(24pt)

{{ cover_letter.greeting | typst_escape }}

#v(12pt)

{% for paragraph in cover_letter.paragraphs %}
{{ paragraph | typst_escape }}

#v(12pt)
{% endfor %}

#v(8pt)

{{ cover_letter.closing | typst_escape }},

#v(24pt)

{{ cover_letter.signature | typst_escape }}
