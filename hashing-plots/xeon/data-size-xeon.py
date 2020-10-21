# coding: utf-8

# In[19]:

import numpy as np
import sys
import pandas as pd
import numpy
import itertools
import toyplot
import toyplot.pdf
#get_ipython().run_line_magic('matplotlib', 'inline')


filename = sys.argv[1]
col_0_name = "numPairs"
y_label_0 = "Insertion Throughput (billions pairs/sec)"
x_label_0 = "Billions of Key-Value pairs"
title_0 = "Xeon Gold Insertions"
y_label_1 = "Query Throughput (billions pairs/sec)"
x_label_1 = "Billions of Key-Value pairs"
title_1 = "Xeon Gold Querying"


df = pd.read_csv(filename, header=None, names=['numPairs0', 'hfInsert00', 'hfQuery00', 'cuInsert00', 'cuQuery00', 'thSort0', 'thSearch0', 'hfInsert01', 'hfQuery01', 'cuInsert01', 'cuQuery01'])


df['hfInsert00'] = df.numPairs0/df.hfInsert00
df['cuInsert00'] = df.numPairs0/df.cuInsert00
df['thSort0'] = df.numPairs0/df.thSort0
df['hfInsert01'] = df.numPairs0/df.hfInsert01
df['cuInsert01'] = df.numPairs0/df.cuInsert01
df['hfQuery00'] = df.numPairs0/df.hfQuery00
df['cuQuery00'] = df.numPairs0/df.cuQuery00
df['thSearch0'] = df.numPairs0/df.thSearch0
df['hfQuery01'] = df.numPairs0/df.hfQuery01
df['cuQuery01'] = df.numPairs0/df.cuQuery01

df['numPairs0'] = df.numPairs0/1000000

colors = ["green", "steelblue", "darkred",
          "green", "steelblue"]
markers = ["o", "--", "^",
           "o", "--"]
exts = ["", "2"]

def GetStyle(toplot) :
  style = {}
  keys = algos.keys()
  for key in keys:
    if key in toplot :
      index = algos.get(key)
      style["stroke"] = colors[index]
    if "01" in toplot or "11" in toplot :
      style["stroke-dasharray"] = "2, 2"
  return style


canvas = toyplot.Canvas('3.5in', '6in')

#Plot 1: Xeon Gold Insertions
algos = {"hfInsert00":0, "cuInsert00":1, "thSort0":2,
         "hfInsert01":3, "cuInsert01":4}
legends = {"hfInsert00":"HashFight,1.03", \
           "cuInsert00":"TBB-Map,1.03", \
           "thSort0":"Thrust-Sort", \
           "hfInsert01":"HashFight,1.50", \
           "cuInsert01":"TBB-Map,1.50"}

axes = canvas.cartesian(label = title_0,
                        xlabel = 'Key-Value Pairs (1M)',
                        ylabel = 'Insert Throughput (1M pairs/sec)',
                        grid = (2,1,0),
                        margin = (40,50,45))
axes.label.style = {"font-size":"12px"}
#axes.label.location = "above"
#axes.label.offset = "-20px"
axes.x.label.style = {"font-size":"8px"}
axes.y.label.style = {"font-size":"8px"}
axes.x.ticks.show = True
axes.y.ticks.show = True
axes.y.ticks.labels.angle = -90
axes.x.ticks.far = "4px"
axes.y.ticks.far = "4px"
axes.y.label.location = "above"
axes.y.label.offset = "30px"
axes.x.ticks.locator = toyplot.locator.Explicit([50, 500, 1000, 1450])
axes.y.ticks.locator = toyplot.locator.Explicit([0, 10, 100, 200, 250, 300])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    t = df.numPairs0.values
    m = t.argmax()
    x = [c for c in itertools.islice(t, 0, m)]
    t = numpy.array(data)
    y = [c for c in itertools.islice(t, 0, m)]
    plotstyle = GetStyle(toplot)
    label = legends.get(toplot)
    thisMarker = markers[algos.get(algo)]
    thisStroke = plotstyle.get("stroke")
    print len(x)
    print len(y)
    axes.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "middle"
    if "cuInsert00" in algo:
    	vertalign = "top"
	if "cuInsert01" in algo:
		vertalign = "bottom"
    y_end = np.count_nonzero(~np.isnan(y)) - 1
    axes.text(x[y_end], y[y_end], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"7.5px", \
                                          "-toyplot-vertical-align":vertalign})

#Plot 2: Xeon Gold Queries
algos = {"hfQuery00":0, "cuQuery00":1, "thSearch0":2,
         "hfQuery01":3, "cuQuery01":4}
legends = {"hfQuery00":"HashFight,1.03", \
           "cuQuery00":"TBB-Map,1.03", \
           "thSearch0":"Thrust-Search", \
           "hfQuery01":"HashFight,1.50", \
           "cuQuery01":"TBB-Map,1.50"}

axes2 = canvas.cartesian(label = title_1,
                        xlabel = 'Key-Value Pairs (1M)',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid=(2,1,1),
                        margin = (40,50,50))

#axes2.label.location = "above"
#axes2.label.offset = "-20px"
axes2.label.style = {"font-size":"12px"}
axes2.x.label.style = {"font-size":"8px"}
axes2.y.label.style = {"font-size":"8px"}
axes2.x.ticks.show = True
axes2.y.ticks.show = True
axes2.y.ticks.labels.angle = -90
axes2.x.ticks.far = "4px"
axes2.y.ticks.far = "4px"
axes2.y.label.location = "above"
axes2.y.label.offset = "30px"

axes2.x.ticks.locator = toyplot.locator.Explicit([50, 500, 1000, 1450])
axes2.y.ticks.locator = toyplot.locator.Explicit([0, 25, 100, 200, 300, 400, 500])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    t = df.numPairs0.values
    m = t.argmax()
    x = [c for c in itertools.islice(t, 0, m)]
    t = numpy.array(data)
    y = [c for c in itertools.islice(t, 0, m)]
    plotstyle = GetStyle(toplot)
    label = legends.get(toplot)
    thisMarker = markers[algos.get(algo)]
    thisStroke = plotstyle.get("stroke")
    axes2.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes2._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "middle"
    if "cuQuery00" in algo:
    	vertalign = "top"
	if "cuQuery01" in algo:
		vertalign = "bottom"

    y_end = np.count_nonzero(~np.isnan(y)) - 1
    axes2.text(x[y_end], y[y_end], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"7.5px", \
                                          "-toyplot-vertical-align":vertalign})


canvas.aspect = 'fit-range'
toyplot.pdf.render(canvas, "data-size-xeon.pdf")
