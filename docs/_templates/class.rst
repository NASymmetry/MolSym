{{ fullname | escape | underline}}

{{ module }}

{{ objname }}

{% block attributes %} {% if attributes %} .. rubric:: Attributes

{% for item in attributes %}
~{{ fullname }}.{{ item }}

{%- endfor %} {% endif %} {% endblock %}

{% block methods %} {% if methods %} .. rubric:: Methods

{% for item in methods %}
{%- if item != '__init__' %} ~{{ fullname }}.{{ item }} {%- endif -%}

{%- endfor %} {% endif %} {% endblock %}