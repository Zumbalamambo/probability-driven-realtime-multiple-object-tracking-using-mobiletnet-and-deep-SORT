import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from collections import OrderedDict

from bokeh.plotting import output_file, figure, show
from bokeh.layouts import gridplot
from bokeh.models import LinearAxis, Range1d, Plot, DataRange1d, ColumnDataSource, HoverTool, TapTool, OpenURL
import bokeh.models as bm
from bokeh.palettes import Viridis6
from bokeh.models.tools import BoxSelectTool

class MOT_eval(object):
    def __init__(self, data):
        self.data = data
    
    def evaluation(self):
        raise NotImplementedError
    
    def normalization(self, data, max_val, min_val, method='max_min', **kwargs):
        new_data = []
        temp_data = []
        for i, d in enumerate(data):
            if(method == 'max_min'):
                new_data.append((d - min_val) / (max_val - min_val))
            elif(method == 'max_min_diff'):
                if(i == 0):
                    new_data.append((d - min_val) / (max_val - min_val))
                else:
                    d = data[i] - data[i-1]
                    new_data.append((d - min_val) / (max_val - min_val))
            elif(method == 'diff_of_max_min'):
                temp_data.append((d - min_val) / (max_val - min_val))
        
        if(method == 'diff_of_max_min'):
            for i, temp in enumerate(temp_data):
                if(i == 0):
                    continue
                else:
                    new_data.append(temp_data[i] - temp_data[i-1])

        return new_data

    def standardization(self, data, method='sigmoid', **kwargs):
        new_data = []
        for i, d in enumerate(data):
            if(method == 'sigmoid'):
                new_data.append(1 / (1 + np.exp(-d)))
            elif(method == 'zscore'):
                new_data.append((d - kwargs['mean']) / kwargs['std'])
            else:
                new_data.append(d)
        return new_data

    def euclidean_distance(self, X, Y, method='euclidean', vis=False):
        distances = list()
        for x, y in zip(X, Y):
            print(x,y)
            if(method == 'euclidean'):
                distances.append(np.sqrt(x**2 +  y**2))
            elif(method == 'manhattan'):
                distances.append(np.abs(x) + np.abs(y))
            elif(method == 'div'):
                distances.append(x / y)
            else:
                raise NotImplementedError
        return distances

    def increment_and_decrement(self, X, Y):
        increment = []
        decrement = []
        for i in range(len(X)):
            if(i % 2 == 0 and i != 0):
                increment.append((X[i] - X[i-1]) / X[i])
                decrement.append((Y[i] - Y[i-1]) / Y[i])
        return increment, decrement, [increment[i] / decrement[i] for i in range(len(increment))]

    def visualization(self):
        # Data prepare
        METRICS_MAPPING = {
            'skip_frame': ['vanilla', 'skip1', 'skip2', 'skip3', 'skip4', 'skip5', 'skip6', 'skip7', 'skip8', 'skip9', 'skip10'],
            'downsampling': ['vanilla', 'skip1_downsampling', 'skip2_downsampling', 'skip3_downsampling', 'skip4_downsampling', 'skip5_downsampling', 'skip6_downsampling', 'skip7_downsampling', 'skip8_downsampling', 'skip9_downsampling', 'skip10_downsampling'],
            'prob_driven': ['vanilla', 'skip1_prob', 'skip2_prob', 'skip3_prob', 'skip4_prob', 'skip5_prob', 'skip6_prob', 'skip7_prob', 'skip8_prob', 'skip9_prob', 'skip10_prob'],
            'downsampling_with_prob_driven': ['vanilla', 'skip1_downsampling_prob', 'skip2_downsampling_prob', 'skip3_downsampling_prob', 'skip4_downsampling_prob', 'skip5_downsampling_prob', 'skip6_downsampling_prob', 'skip7_downsampling_prob', 'skip8_downsampling_prob', 'skip9_downsampling_prob', 'skip10_downsampling_prob'],
            'vanilla': ['vanilla'],
            'skip1': ['skip1', 'skip1_prob'],
            'skip2': ['skip2', 'skip2_prob'],
            'skip3': ['skip3', 'skip3_prob'],
            'skip4': ['skip4', 'skip4_prob'],
            'skip5': ['skip5', 'skip5_prob'],
            'skip6': ['skip6', 'skip6_prob'],
            'skip7': ['skip7', 'skip7_prob'],
            'skip8': ['skip8', 'skip8_prob'],
            'skip9': ['skip9', 'skip9_prob'],
            'skip10': ['skip10', 'skip10_prob'],
        }
        METRICS_MAPPING['all'] = METRICS_MAPPING['vanilla'] + METRICS_MAPPING['skip1'] + METRICS_MAPPING['skip2'] + METRICS_MAPPING['skip3'] + METRICS_MAPPING['skip4'] + METRICS_MAPPING['skip5'] + METRICS_MAPPING['skip6'] + METRICS_MAPPING['skip7'] + METRICS_MAPPING['skip8'] + METRICS_MAPPING['skip9'] + METRICS_MAPPING['skip10']

        # Keys = yolov3, mobilenet ssd, squeeze net 1.0
        keys = list(self.data.keys())
        
        FPS_data = {}
        MOTA_data = {}
        for key, value in self.data.items():
            FPS_data[key] = {}
            FPS_data[key]['skip_frame'] = []
            FPS_data[key]['downsampling'] = []
            FPS_data[key]['prob_driven'] = []
            FPS_data[key]['downsampling_with_prob_driven'] = []
            FPS_data[key]['vanilla'] = []
            FPS_data[key]['skip1'] = []
            FPS_data[key]['skip2'] = []
            FPS_data[key]['skip3'] = []
            FPS_data[key]['skip4'] = []
            FPS_data[key]['skip5'] = []
            FPS_data[key]['skip6'] = []
            FPS_data[key]['skip7'] = []
            FPS_data[key]['skip8'] = []
            FPS_data[key]['skip9'] = []
            FPS_data[key]['skip10'] = []
            FPS_data[key]['all'] = []
            #FPS_data[key]['color'] = Viridis6
            
            MOTA_data[key] = {}
            MOTA_data[key]['skip_frame'] = []
            MOTA_data[key]['downsampling'] = []
            MOTA_data[key]['prob_driven'] = []
            MOTA_data[key]['downsampling_with_prob_driven'] = []
            MOTA_data[key]['vanilla'] = []
            MOTA_data[key]['skip1'] = []
            MOTA_data[key]['skip2'] = []
            MOTA_data[key]['skip3'] = []
            MOTA_data[key]['skip4'] = []
            MOTA_data[key]['skip5'] = []
            MOTA_data[key]['skip6'] = []
            MOTA_data[key]['skip7'] = []
            MOTA_data[key]['skip8'] = []
            MOTA_data[key]['skip9'] = []
            MOTA_data[key]['skip10'] = []
            MOTA_data[key]['all'] = []
            #MOTA_data[key]['color'] = Viridis6
            for algorithm, metrics in value.items():
                if(algorithm in METRICS_MAPPING['skip_frame']):
                    FPS_data[key]['skip_frame'].append(metrics['FPS'])
                    MOTA_data[key]['skip_frame'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['downsampling']):
                    FPS_data[key]['downsampling'].append(metrics['FPS'])
                    MOTA_data[key]['downsampling'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['prob_driven']):
                    FPS_data[key]['prob_driven'].append(metrics['FPS'])
                    MOTA_data[key]['prob_driven'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['downsampling_with_prob_driven']):
                    FPS_data[key]['downsampling_with_prob_driven'].append(metrics['FPS'])
                    MOTA_data[key]['downsampling_with_prob_driven'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['all']):
                    FPS_data[key]['all'].append(metrics['FPS'])
                    MOTA_data[key]['all'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['vanilla']):
                    FPS_data[key]['vanilla'].append(metrics['FPS'])
                    MOTA_data[key]['vanilla'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip1']):
                    FPS_data[key]['skip1'].append(metrics['FPS'])
                    MOTA_data[key]['skip1'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip2']):
                    FPS_data[key]['skip2'].append(metrics['FPS'])
                    MOTA_data[key]['skip2'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip3']):
                    FPS_data[key]['skip3'].append(metrics['FPS'])
                    MOTA_data[key]['skip3'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip4']):
                    FPS_data[key]['skip4'].append(metrics['FPS'])
                    MOTA_data[key]['skip4'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip5']):
                    FPS_data[key]['skip5'].append(metrics['FPS'])
                    MOTA_data[key]['skip5'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip6']):
                    FPS_data[key]['skip6'].append(metrics['FPS'])
                    MOTA_data[key]['skip6'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip7']):
                    FPS_data[key]['skip7'].append(metrics['FPS'])
                    MOTA_data[key]['skip7'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip8']):
                    FPS_data[key]['skip8'].append(metrics['FPS'])
                    MOTA_data[key]['skip8'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip9']):
                    FPS_data[key]['skip9'].append(metrics['FPS'])
                    MOTA_data[key]['skip9'].append(metrics['MOTA'])
                if(algorithm in METRICS_MAPPING['skip10']):
                    FPS_data[key]['skip10'].append(metrics['FPS'])
                    MOTA_data[key]['skip10'].append(metrics['MOTA'])
        
        # Normalization
        colors = ['red', 'green', 'blue', 'purple', 'orange']
        KEY_INDEX = 2
        NORM_METHOD = 'diff_of_max_min'

        MAX_MOTA = max(MOTA_data[keys[KEY_INDEX]]['all'])
        MIN_MOTA = min(MOTA_data[keys[KEY_INDEX]]['all'])
        MAX_FPS = max(FPS_data[keys[KEY_INDEX]]['all'])
        MIN_FPS = min(FPS_data[keys[KEY_INDEX]]['all'])

        all_MOTA_arr = np.array(MOTA_data[keys[KEY_INDEX]]['all'])
        all_FPS_arr = np.array(FPS_data[keys[KEY_INDEX]]['all'])
        mean_MOTA = np.mean(all_MOTA_arr, axis=0)
        std_MOTA = np.std(all_MOTA_arr, axis=0)

        mean_FPS = np.mean(all_FPS_arr, axis=0)
        std_FPS = np.std(all_FPS_arr, axis=0)


        # MOTA vs. FPS 
        #print(self.increment_and_decrement(MOTA_data[keys[KEY_INDEX]]['all'], FPS_data[keys[KEY_INDEX]]['all']))

        TOOLS = 'hover, pan,wheel_zoom,reset,save'

        plot_info = ['skip_frame', 'downsampling', 'prob_driven', 'downsampling_with_prob_driven']
        color_info = ['red', 'green', 'blue', 'purple']

        KEY_INDEX = 0
        p_yolov3_mota = figure(title = "YOLOv3 MOTA vs. FPS", tools=[TOOLS])
        for i, info in enumerate(plot_info):
            yolov3_mota_source = ColumnDataSource(data=dict(
                mota=MOTA_data[keys[KEY_INDEX]][info],
                fps=FPS_data[keys[KEY_INDEX]][info],
                desc=[info] * len(MOTA_data[keys[KEY_INDEX]][info]),
                legend=[info] * len(MOTA_data[keys[KEY_INDEX]][info]),
            ))

            p_yolov3_mota.circle('fps', 'mota',source=yolov3_mota_source, legend='legend', fill_color="white", size=4, color=color_info[i])
            p_yolov3_mota.line('fps', 'mota', source=yolov3_mota_source, legend='legend', line_width=4, line_color=color_info[i], line_alpha=0.6, hover_line_color=color_info[i], hover_line_alpha=0.9) 

            p_yolov3_mota.legend.location = "top_right"
            p_yolov3_mota.legend.click_policy="hide"
            p_yolov3_mota.yaxis.axis_label = "MOTA"
            p_yolov3_mota.xaxis.axis_label = "FPS"
        hover = p_yolov3_mota.select(dict(type=HoverTool))
        hover.tooltips = [("FPS", "@fps"),("MOTA", "@mota")]
        hover.mode = 'mouse'

        KEY_INDEX = 1
        p_mobilenetssd_mota = figure(title = "MOBILENET SSD MOTA vs. FPS", tools=[TOOLS])
        for i, info in enumerate(plot_info):
            yolov3_mota_source = ColumnDataSource(data=dict(
                mota=MOTA_data[keys[KEY_INDEX]][info],
                fps=FPS_data[keys[KEY_INDEX]][info],
                desc=[info] * len(MOTA_data[keys[KEY_INDEX]][info]),
                legend=[info] * len(MOTA_data[keys[KEY_INDEX]][info]),
            ))

            p_mobilenetssd_mota.circle('fps', 'mota',source=yolov3_mota_source, legend='legend', fill_color="white", size=4, color=color_info[i])
            p_mobilenetssd_mota.line('fps', 'mota', source=yolov3_mota_source, legend='legend', line_width=4, line_color=color_info[i], line_alpha=0.6, hover_line_color=color_info[i], hover_line_alpha=0.9) 

            p_mobilenetssd_mota.legend.location = "top_right"
            p_mobilenetssd_mota.legend.click_policy="hide"
            p_mobilenetssd_mota.yaxis.axis_label = "MOTA"
            p_mobilenetssd_mota.xaxis.axis_label = "FPS"
        hover = p_mobilenetssd_mota.select(dict(type=HoverTool))
        hover.tooltips = [("FPS", "@fps"),("MOTA", "@mota")]
        hover.mode = 'mouse'
        
        KEY_INDEX = 2
        p_squeezenetv10_mota = figure(title = "SQUEEZENET V1.0 MOTA vs. FPS", tools=[TOOLS])
        for i, info in enumerate(plot_info):
            yolov3_mota_source = ColumnDataSource(data=dict(
                mota=MOTA_data[keys[KEY_INDEX]][info],
                fps=FPS_data[keys[KEY_INDEX]][info],
                desc=[info] * len(MOTA_data[keys[KEY_INDEX]][info]),
                legend=[info] * len(MOTA_data[keys[KEY_INDEX]][info]),
            ))

            p_squeezenetv10_mota.circle('fps', 'mota',source=yolov3_mota_source, legend='legend', fill_color="white", size=4, color=color_info[i])
            p_squeezenetv10_mota.line('fps', 'mota', source=yolov3_mota_source, legend='legend', line_width=4, line_color=color_info[i], line_alpha=0.6, hover_line_color=color_info[i], hover_line_alpha=0.9) 

            p_squeezenetv10_mota.legend.location = "top_right"
            p_squeezenetv10_mota.legend.click_policy="hide"
            p_squeezenetv10_mota.yaxis.axis_label = "MOTA"
            p_squeezenetv10_mota.xaxis.axis_label = "FPS"
        hover = p_mobilenetssd_mota.select(dict(type=HoverTool))
        hover.tooltips = [("FPS", "@fps"),("MOTA", "@mota")]
        hover.mode = 'mouse'

        show(gridplot([[p_yolov3_mota], [p_mobilenetssd_mota], [p_squeezenetv10_mota]], plot_width=1000, plot_height=600))


        """
        # MOTA and FPS comparison plot

        ## YOLOV3
        TOOLS = 'pan,wheel_zoom,reset,save'

        yolov3_mota_source = ColumnDataSource(data=dict(
            x=[[i for i in list(range(4))] for j in range(4)],
            y=[MOTA_data[keys[0]]['skip_frame'],MOTA_data[keys[0]]['downsampling'],MOTA_data[keys[0]]['prob_driven'],MOTA_data[keys[0]]['downsampling_with_prob_driven']],
            desc=list(MOTA_data[keys[0]].keys()),
            color=['red', 'green', 'blue', 'purple'],
            legend=list(MOTA_data[keys[0]].keys()),
        ))

        yolov3_mota_hover = HoverTool(tooltips=[
                            ("index", "$index"),
                            ("MOTA", "$y"),
                            ("desc", "@desc"),], 
                            mode='mouse',
        )

        p_yolov3_mota = figure(title='YOLOv3 MOTA', tools=[TOOLS, yolov3_mota_hover])
        p_yolov3_mota.multi_line('x', 'y', legend="legend", line_width=4, line_color='color', line_alpha=0.6, hover_line_color='color', hover_line_alpha=1.0, source=yolov3_mota_source)
        p_yolov3_mota.legend.location = "top_right"
        p_yolov3_mota.yaxis.axis_label = "MOTA"

        yolov3_fps_source = ColumnDataSource(data=dict(
            x=[[i for i in list(range(4))] for j in range(4)],
            y=[FPS_data[keys[0]]['skip_frame'],FPS_data[keys[0]]['downsampling'],FPS_data[keys[0]]['prob_driven'],FPS_data[keys[0]]['downsampling_with_prob_driven']],
            desc=list(FPS_data[keys[0]].keys()),
            color=['red', 'green', 'blue', 'purple'],
            legend=list(FPS_data[keys[0]].keys()),
        ))

        yolov3_fps_hover = HoverTool(tooltips=[
                            ("index", "$index"),
                            ("FPS", "$y"),
                            ("desc", "@desc"),], 
                            mode='mouse',
        )

        p_yolov3_fps = figure(title='YOLOv3 FPS', tools=[TOOLS, yolov3_fps_hover])
        p_yolov3_fps.multi_line('x', 'y', legend="legend", line_width=4, line_color='color', line_alpha=0.6, hover_line_color='color', hover_line_alpha=1.0, source=yolov3_fps_source)
        p_yolov3_fps.legend.location = "top_right"
        p_yolov3_fps.yaxis.axis_label = "FPS"


        ## Mobilenet SSD
        mobilenet_mota_source = ColumnDataSource(data=dict(
            x=[[i for i in list(range(4))] for j in range(4)],
            y=[MOTA_data[keys[1]]['skip_frame'],MOTA_data[keys[1]]['downsampling'],MOTA_data[keys[1]]['prob_driven'],MOTA_data[keys[1]]['downsampling_with_prob_driven']],
            desc=list(MOTA_data[keys[1]].keys()),
            color=['red', 'green', 'blue', 'purple'],
            legend=list(MOTA_data[keys[1]].keys()),
        ))

        mobilenet_mota_hover = HoverTool(tooltips=[
                            ("index", "$index"),
                            ("MOTA", "$y"),
                            ("desc", "@desc"),], 
                            mode='mouse',
        )

        p_mobilenet_mota = figure(title='Mobilenet MOTA', tools=[TOOLS, mobilenet_mota_hover])
        p_mobilenet_mota.multi_line('x', 'y', legend="legend", line_width=4, line_color='color', line_alpha=0.6, hover_line_color='color', hover_line_alpha=1.0, source=mobilenet_mota_source)
        p_mobilenet_mota.legend.location = "top_right"
        p_mobilenet_mota.yaxis.axis_label = "MOTA"

        mobilenet_fps_source = ColumnDataSource(data=dict(
            x=[[i for i in list(range(4))] for j in range(4)],
            y=[FPS_data[keys[1]]['skip_frame'],FPS_data[keys[1]]['downsampling'],FPS_data[keys[1]]['prob_driven'],FPS_data[keys[1]]['downsampling_with_prob_driven']],
            desc=list(FPS_data[keys[1]].keys()),
            color=['red', 'green', 'blue', 'purple'],
            legend=list(FPS_data[keys[1]].keys()),
        ))

        mobilenet_fps_hover = HoverTool(tooltips=[
                            ("index", "$index"),
                            ("FPS", "$y"),
                            ("desc", "@desc"),], 
                            mode='mouse',
        )

        p_mobilenet_fps = figure(title='Mobilenet FPS', tools=[TOOLS, mobilenet_fps_hover])
        p_mobilenet_fps.multi_line('x', 'y', legend="legend", line_width=4, line_color='color', line_alpha=0.6, hover_line_color='color', hover_line_alpha=1.0, source=mobilenet_fps_source)
        p_mobilenet_fps.legend.location = "top_right"
        p_mobilenet_fps.yaxis.axis_label = "FPS"

        ## Squeezenet 1.0
        squeezenetv1_0_mota_source = ColumnDataSource(data=dict(
            x=[[i for i in list(range(4))] for j in range(4)],
            y=[MOTA_data[keys[2]]['skip_frame'],MOTA_data[keys[2]]['downsampling'],MOTA_data[keys[2]]['prob_driven'],MOTA_data[keys[2]]['downsampling_with_prob_driven']],
            desc=list(MOTA_data[keys[2]].keys()),
            color=['red', 'green', 'blue', 'purple'],
            legend=list(MOTA_data[keys[2]].keys()),
        ))

        squeezenetv1_0_mota_hover = HoverTool(tooltips=[
                                    ("index", "$index"),
                                    ("MOTA", "$y"),
                                    ("desc", "@desc"),], 
                                    mode='mouse',
        )

        p_squeezenetv1_0_mota = figure(title='SqueezeNet v1.0 MOTA', tools=[TOOLS, squeezenetv1_0_mota_hover])
        p_squeezenetv1_0_mota.multi_line('x', 'y', legend="legend", line_width=4, line_color='color', line_alpha=0.6, hover_line_color='color', hover_line_alpha=1.0, source=squeezenetv1_0_mota_source)
        p_squeezenetv1_0_mota.legend.location = "top_right"
        p_squeezenetv1_0_mota.yaxis.axis_label = "MOTA"

        squeezenetv1_0_fps_source = ColumnDataSource(data=dict(
            x=[[i for i in list(range(4))] for j in range(4)],
            y=[FPS_data[keys[2]]['skip_frame'],FPS_data[keys[2]]['downsampling'],FPS_data[keys[2]]['prob_driven'],FPS_data[keys[2]]['downsampling_with_prob_driven']],
            desc=list(FPS_data[keys[2]].keys()),
            color=['red', 'green', 'blue', 'purple'],
            legend=list(FPS_data[keys[2]].keys()),
        ))

        squeezenetv1_0_fps_hover = HoverTool(tooltips=[
                                            ("index", "$index"),
                                            ("FPS", "$y"),
                                            ("desc", "@desc"),], 
                                            mode='mouse',
        )

        p_squeezenetv1_0_fps = figure(title='SqueezeNet v1.0 FPS', tools=[TOOLS, squeezenetv1_0_fps_hover])
        p_squeezenetv1_0_fps.multi_line('x', 'y', legend="legend", line_width=4, line_color='color', line_alpha=0.6, hover_line_color='color', hover_line_alpha=1.0, source=squeezenetv1_0_fps_source)
        p_squeezenetv1_0_fps.legend.location = "top_right"
        p_squeezenetv1_0_fps.yaxis.axis_label = "FPS"



        show(gridplot([[p_yolov3_mota, p_yolov3_fps], [p_mobilenet_mota, p_mobilenet_fps], [p_squeezenetv1_0_mota, p_squeezenetv1_0_fps]], plot_width=600, plot_height=600))
        """

if __name__ == '__main__':
    data = {
        'yolov3_tiny': {'vanilla':                  {'MOTA':0.336, 'IDsw':635, 'FPS':10.91483091271569},
                        'vanilla_downsampling':     {'MOTA':0.328, 'IDsw':651, 'FPS':13.502331711038652},
                        'skip1':                    {'MOTA':0.306, 'IDsw':570, 'FPS':19.293153879329523},
                        'skip1_downsampling':       {'MOTA':0.299, 'IDsw':577, 'FPS':21.16659516265331},
                        'skip1_prob':               {'MOTA':0.322, 'IDsw':594, 'FPS':15.250728432641399},
                        'skip1_downsampling_prob':  {'MOTA':0.317, 'IDsw':601, 'FPS':17.246271081266418},
                        'skip2':                    {'MOTA':0.262, 'IDsw':581, 'FPS':24.380912639561338},
                        'skip2_downsampling':       {'MOTA':0.255, 'IDsw':538, 'FPS':26.39487985590097},
                        'skip2_prob':               {'MOTA':0.304, 'IDsw':581, 'FPS':17.273034259482532},
                        'skip2_downsampling_prob':  {'MOTA':0.299, 'IDsw':561, 'FPS':19.306497472601286},
                        'skip3':                    {'MOTA':0.214, 'IDsw':581, 'FPS':28.105401141453655},
                        'skip3_downsampling':       {'MOTA':0.212, 'IDsw':538, 'FPS':30.1861545267122},
                        'skip3_prob':               {'MOTA':0.285, 'IDsw':581, 'FPS':19.220862812101863},
                        'skip3_downsampling_prob':  {'MOTA':0.282, 'IDsw':561, 'FPS':21.497436502860783},
                        'skip4':                    {'MOTA':0.173, 'IDsw':858, 'FPS':30.91609208868686},
                        'skip4_downsampling':       {'MOTA':0.172, 'IDsw':826, 'FPS':32.97000749116024},
                        'skip4_prob':               {'MOTA':0.269, 'IDsw':778, 'FPS':20.452854541630085},
                        'skip4_downsampling_prob':  {'MOTA':0.263, 'IDsw':721, 'FPS':23.33499897346363},


                        'skip4':                    {'MOTA':0.138, 'IDsw':858, 'FPS':33.1634016095078},
                        'skip4_downsampling':       {'MOTA':0.136, 'IDsw':826, 'FPS':35.495408777649196},
                        'skip4_prob':               {'MOTA':0.203, 'IDsw':778, 'FPS':27.908364645956375},
                        'skip4_downsampling_prob':  {'MOTA':0.189, 'IDsw':721, 'FPS':30.19416683885809},
                        'skip5':                    {'MOTA':0.114, 'IDsw':858, 'FPS':35.37700094177629},
                        'skip5_downsampling':       {'MOTA':0.110, 'IDsw':826, 'FPS':37.192013066238104},
                        'skip5_prob':               {'MOTA':0.178, 'IDsw':778, 'FPS':28.97554816233987},
                        'skip5_downsampling_prob':  {'MOTA':0.172, 'IDsw':721, 'FPS':31.403710279478073},
                        'skip7':                    {'MOTA':0.088, 'IDsw':809, 'FPS':36.69362283873649},
                        'skip7_downsampling':       {'MOTA':0.085, 'IDsw':821, 'FPS':38.73970200190891},
                        'skip7_prob':               {'MOTA':0.159, 'IDsw':758, 'FPS':30.685951335910243},
                        'skip7_downsampling_prob':  {'MOTA':0.150, 'IDsw':760, 'FPS':33.450852756265625},
                        'skip8':                    {'MOTA':0.068, 'IDsw':809, 'FPS':38.141400874916755},
                        'skip8_downsampling':       {'MOTA':0.062, 'IDsw':821, 'FPS':40.26659519478607},
                        'skip8_prob':               {'MOTA':0.136, 'IDsw':758, 'FPS':31.7064167890517},
                        'skip8_downsampling_prob':  {'MOTA':0.132, 'IDsw':760, 'FPS':34.11690588833806},
                        'skip9':                    {'MOTA':0.052, 'IDsw':809, 'FPS':39.11637415522933},
                        'skip9_downsampling':       {'MOTA':0.050, 'IDsw':821, 'FPS':41.54897992498731},
                        'skip9_prob':               {'MOTA':0.115, 'IDsw':758, 'FPS':34.267741847251465},
                        'skip9_downsampling_prob':  {'MOTA':0.106, 'IDsw':760, 'FPS':36.149916359826285},
                        'skip10':                   {'MOTA':0.029, 'IDsw':809, 'FPS':40.17551909751923},
                        'skip10_downsampling':      {'MOTA':0.029, 'IDsw':821, 'FPS':42.46852955687042},
                        'skip10_prob':              {'MOTA':0.095, 'IDsw':758, 'FPS':34.33952922260652},
                        'skip10_downsampling_prob': {'MOTA':0.091, 'IDsw':760, 'FPS':36.742833768287326},
        },

        'Mobilenetv1': {'vanilla':                  {'MOTA':0.190, 'IDsw':577, 'FPS':9.587366376720645   },
                        'vanilla_downsampling':     {'MOTA':0.175, 'IDsw':647, 'FPS':9.604095582070949   },
                        'skip1':                    {'MOTA':0.176, 'IDsw':520, 'FPS':16.252922740164827   },
                        'skip1_downsampling':       {'MOTA':0.166, 'IDsw':569, 'FPS':16.40143629589119   },
                        'skip1_prob':               {'MOTA':0.180, 'IDsw':523, 'FPS':14.637242356905144 },
                        'skip1_downsampling_prob':  {'MOTA':0.168, 'IDsw':573, 'FPS':15.00819154341061 },
                        'skip2':                    {'MOTA':0.156, 'IDsw':494, 'FPS':21.112713136898233 },
                        'skip2_downsampling':       {'MOTA':0.144, 'IDsw':530, 'FPS':21.599045498123022  },
                        'skip2_prob':               {'MOTA':0.165, 'IDsw':503, 'FPS':18.271082139494112 },
                        'skip2_downsampling_prob':  {'MOTA':0.155, 'IDsw':553, 'FPS':18.940480585788073 },
                        'skip3':                    {'MOTA':0.125, 'IDsw':556, 'FPS':25.031793392567     },
                        'skip3_downsampling':       {'MOTA':0.115, 'IDsw':541, 'FPS':25.69524476916254  },
                        'skip3_prob':               {'MOTA':0.143, 'IDsw':551, 'FPS':20.73637006639062   },
                        'skip3_downsampling_prob':  {'MOTA':0.137, 'IDsw':541, 'FPS':21.838736638293103 },
                        'skip4':                    {'MOTA':0.107, 'IDsw':569, 'FPS':28.248501152598728  },
                        'skip4_downsampling':       {'MOTA':0.095, 'IDsw':557, 'FPS':28.92964870730287   },
                        'skip4_prob':               {'MOTA':0.132, 'IDsw':552, 'FPS':22.9520797318525    },
                        'skip4_downsampling_prob':  {'MOTA':0.121, 'IDsw':567, 'FPS':24.037214225739753 },
                        'skip5':                    {'MOTA':0.090, 'IDsw':607, 'FPS':30.847425214202232  },
                        'skip5_downsampling':       {'MOTA':0.083, 'IDsw':632, 'FPS':31.919496472324205  },
                        'skip5_prob':               {'MOTA':0.117, 'IDsw':624, 'FPS':25.052836061352583 },
                        'skip5_downsampling_prob':  {'MOTA':0.111, 'IDsw':571, 'FPS':26.00150446850327   },
                        'skip6':                    {'MOTA':0.077, 'IDsw':634, 'FPS':32.77340066189756  },
                        'skip6_downsampling':       {'MOTA':0.060, 'IDsw':589, 'FPS':33.9516748754625     },
                        'skip6_prob':               {'MOTA':0.107, 'IDsw':607, 'FPS':26.31595826272231   },
                        'skip6_downsampling_prob':  {'MOTA':0.100, 'IDsw':555, 'FPS':27.87592524453508   },
                        'skip7':                    {'MOTA':0.046, 'IDsw':608, 'FPS':34.443203535459524  },
                        'skip7_downsampling':       {'MOTA':0.035, 'IDsw':618, 'FPS':35.8327512141732     },
                        'skip7_prob':               {'MOTA':0.087, 'IDsw':592, 'FPS':27.943382730046693  },
                        'skip7_downsampling_prob':  {'MOTA':0.075, 'IDsw':583, 'FPS':29.43941250448985   },
                        'skip8':                    {'MOTA':0.042, 'IDsw':576, 'FPS':36.1859327760888     },
                        'skip8_downsampling':       {'MOTA':0.028, 'IDsw':538, 'FPS':37.96312668466682  },
                        'skip8_prob':               {'MOTA':0.072, 'IDsw':599, 'FPS':30.149078462256718 },
                        'skip8_downsampling_prob':  {'MOTA':0.058, 'IDsw':554, 'FPS':31.716512547097206 },
                        'skip9':                    {'MOTA':0.026, 'IDsw':569, 'FPS':37.851134549878715  },
                        'skip9_downsampling':       {'MOTA':0.016, 'IDsw':566, 'FPS':39.18969878996756  },
                        'skip9_prob':               {'MOTA':0.062, 'IDsw':570, 'FPS':31.81984758148207   },
                        'skip9_downsampling_prob':  {'MOTA':0.051, 'IDsw':566, 'FPS':33.02867515902485   },
                        'skip10':                   {'MOTA':0.008, 'IDsw':566, 'FPS':38.682324486214625  },
                        'skip10_downsampling':      {'MOTA':0.000, 'IDsw':529, 'FPS':40.239209547617996  },
                        'skip10_prob':              {'MOTA':0.052, 'IDsw':555, 'FPS':32.86656775948573   },
                        'skip10_downsampling_prob': {'MOTA':0.034, 'IDsw':547, 'FPS':33.68070048767031   },
        },

        'squeezenetv1_0':   {'vanilla':                 {'MOTA':0.099, 'IDsw':484, 'FPS':21.44935625079553   },
                            'vanilla_downsampling':     {'MOTA':0.094, 'IDsw':433, 'FPS':21.673303061043992 },
                            'skip1':                    {'MOTA':0.093, 'IDsw':470, 'FPS':34.95316996147481   },
                            'skip1_downsampling':       {'MOTA':0.089, 'IDsw':438, 'FPS':36.08590331121377  },
                            'skip1_prob':               {'MOTA':0.093, 'IDsw':475, 'FPS':32.23478373616411     },
                            'skip1_downsampling_prob':  {'MOTA':0.090, 'IDsw':442, 'FPS':33.33157598718744   },
                            'skip2':                    {'MOTA':0.084, 'IDsw':408, 'FPS':44.77171969416171  },
                            'skip2_downsampling':       {'MOTA':0.082, 'IDsw':372, 'FPS':46.824602862962315 },
                            'skip2_prob':               {'MOTA':0.088, 'IDsw':412, 'FPS':40.62085401241766  },
                            'skip2_downsampling_prob':  {'MOTA':0.086, 'IDsw':368, 'FPS':42.75774047975689  },
                            'skip3':                    {'MOTA':0.075, 'IDsw':383, 'FPS':51.622355269656786 },
                            'skip3_downsampling':       {'MOTA':0.073, 'IDsw':338, 'FPS':54.07041484165557  },
                            'skip3_prob':               {'MOTA':0.082, 'IDsw':396, 'FPS':46.27096400687559   },
                            'skip3_downsampling_prob':  {'MOTA':0.080, 'IDsw':349, 'FPS':48.57501635793877   },
                            'skip4':                    {'MOTA':0.068, 'IDsw':328, 'FPS':56.87061749886791   },
                            'skip4_downsampling':       {'MOTA':0.065, 'IDsw':301, 'FPS':60.00883585017237   },
                            'skip4_prob':               {'MOTA':0.078, 'IDsw':343, 'FPS':51.3094441332378   },
                            'skip4_downsampling_prob':  {'MOTA':0.073, 'IDsw':312, 'FPS':54.63552272367222    },
                            'skip5':                    {'MOTA':0.059, 'IDsw':298, 'FPS':61.06823719084773  },
                            'skip5_downsampling':       {'MOTA':0.061, 'IDsw':294, 'FPS':64.56221875850596  },
                            'skip5_prob':               {'MOTA':0.070, 'IDsw':310, 'FPS':54.28577355918863   },
                            'skip5_downsampling_prob':  {'MOTA':0.070, 'IDsw':301, 'FPS':58.4456130845711    },
                            'skip6':                    {'MOTA':0.056, 'IDsw':301, 'FPS':64.21573734646725  },
                            'skip6_downsampling':       {'MOTA':0.054, 'IDsw':284, 'FPS':68.07149014549003  },
                            'skip6_prob':               {'MOTA':0.064, 'IDsw':313, 'FPS':57.72134336646225   },
                            'skip6_downsampling_prob':  {'MOTA':0.060, 'IDsw':297, 'FPS':61.39160527850794    },

                            'skip7':                    {'MOTA':0.044, 'IDsw':281, 'FPS':66.78610024826153  },
                            'skip7_downsampling':       {'MOTA':0.044, 'IDsw':275, 'FPS':70.8977870230824     },
                            'skip7_prob':               {'MOTA':0.059, 'IDsw':313, 'FPS':60.94890175971762   },
                            'skip7_downsampling_prob':  {'MOTA':0.056, 'IDsw':275, 'FPS':64.50761957946298   },
                            'skip8':                    {'MOTA':0.041, 'IDsw':258, 'FPS':70.83954394395735  },
                            'skip8_downsampling':       {'MOTA':0.042, 'IDsw':266, 'FPS':74.81412007335364  },
                            'skip8_prob':               {'MOTA':0.056, 'IDsw':281, 'FPS':64.66455658532034   },
                            'skip8_downsampling_prob':  {'MOTA':0.055, 'IDsw':283, 'FPS':68.37078004680247    },
                            'skip9':                    {'MOTA':0.038, 'IDsw':278, 'FPS':73.28210769652618  },
                            'skip9_downsampling':       {'MOTA':0.038, 'IDsw':268, 'FPS':76.56258851656787  },
                            'skip9_prob':               {'MOTA':0.055, 'IDsw':285, 'FPS':66.95744212183034   },
                            'skip9_downsampling_prob':  {'MOTA':0.050, 'IDsw':277, 'FPS':71.51873091330889   },
                            'skip10':                   {'MOTA':0.035, 'IDsw':236, 'FPS':78.29819637519171  },
                            'skip10_downsampling':      {'MOTA':0.031, 'IDsw':232, 'FPS':79.04940471707698  },
                            'skip10_prob':              {'MOTA':0.051, 'IDsw':246, 'FPS':71.26371855405316    },
                            'skip10_downsampling_prob': {'MOTA':0.048, 'IDsw':230, 'FPS':74.54617349779674   },
        },
    }

    mot_eval = MOT_eval(data)
    mot_eval.visualization()