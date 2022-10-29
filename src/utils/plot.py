from kaleido.scopes.plotly import PlotlyScope
from pathlib import Path

PLOT_CONVERTER = PlotlyScope()

try:
    from plotly.offline import plot
except ModuleNotFoundError:
    pass


def plotly_hide_traces_legend_without_name(fig):
    for trace in fig['data']:
        if trace['name'] is None: trace['showlegend'] = False


def save_plot_with_external_script_link(fig, fpath, auto_open=False):
    raw_html = '<html><head><meta charset="utf-8" />'
    raw_html += '<script type="text/javascript" src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>'
    raw_html += '<body>'
    raw_html += plot(fig, include_plotlyjs=False,
                     output_type='div')
    raw_html += '</body></html>'
    with open(str(fpath), "w") as text_file:
        text_file.write(raw_html)

    if auto_open:
        import webbrowser
        url = "file://" + str(fpath)
        webbrowser.open(url)


def remove_special_characters(path):
    filename = path.name
    for s in [',', '(', ')', ';']:
        filename = filename.replace(s, '')
    return path.parent / filename


def save_multiple_formats(fig, html_path, scale=3, width=1200, height=700):
    png_data = PLOT_CONVERTER.transform(fig, format="png", scale=scale, width=width, height=height)
    html_path = remove_special_characters(html_path)
    Path(html_path).parent.mkdir(exist_ok=True, parents=True)
    png_path = Path(html_path).with_suffix('.png')
    with open(png_path, 'wb') as f:
        f.write(png_data)

    html_path = png_path.with_suffix('.html')
    save_plot_with_external_script_link(fig, html_path)

    return fig
