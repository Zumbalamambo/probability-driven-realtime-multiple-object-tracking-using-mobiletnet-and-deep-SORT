import numpy as np
import pandas as pd
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
    
    def visualization(self):
        # Data prepare
        METRICS_MAPPING = {
            'skip_frame': ['vanilla', 'skip1', 'skip2', 'skip4', 'skip6'],
            'downsampling': ['vanilla', 'skip1_downsampling', 'skip2_downsampling', 'skip4_downsampling', 'skip6_downsampling'],
            'prob_driven': ['vanilla', 'skip1_prob', 'skip2_prob', 'skip4_prob', 'skip6_prob'],
            'downsampling_with_prob_driven': ['vanilla', 'skip1_downsampling_prob', 'skip2_downsampling_prob', 'skip4_downsampling_prob', 'skip6_downsampling_prob'],
        }
        keys = list(self.data.keys())
        
        FPS_data = {}
        MOTA_data = {}
        for key, value in self.data.items():
            FPS_data[key] = {}
            FPS_data[key]['skip_frame'] = []
            FPS_data[key]['downsampling'] = []
            FPS_data[key]['prob_driven'] = []
            FPS_data[key]['downsampling_with_prob_driven'] = []
            #FPS_data[key]['color'] = Viridis6
            
            MOTA_data[key] = {}
            MOTA_data[key]['skip_frame'] = []
            MOTA_data[key]['downsampling'] = []
            MOTA_data[key]['prob_driven'] = []
            MOTA_data[key]['downsampling_with_prob_driven'] = []
            #MOTA_data[key]['color'] = Viridis6
            for algrithm, metrics in value.items():
                if(algrithm in METRICS_MAPPING['skip_frame']):
                    FPS_data[key]['skip_frame'].append(metrics['FPS'])
                    MOTA_data[key]['skip_frame'].append(metrics['MOTA'])
                if(algrithm in METRICS_MAPPING['downsampling']):
                    FPS_data[key]['downsampling'].append(metrics['FPS'])
                    MOTA_data[key]['downsampling'].append(metrics['MOTA'])
                if(algrithm in METRICS_MAPPING['prob_driven']):
                    FPS_data[key]['prob_driven'].append(metrics['FPS'])
                    MOTA_data[key]['prob_driven'].append(metrics['MOTA'])
                if(algrithm in METRICS_MAPPING['downsampling_with_prob_driven']):
                    FPS_data[key]['downsampling_with_prob_driven'].append(metrics['FPS'])
                    MOTA_data[key]['downsampling_with_prob_driven'].append(metrics['MOTA'])

        # Constrcut Bokeh

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

if __name__ == '__main__':
    data = {
        'yolov3_tiny': {'vanilla':                  {'MOTA':0.336, 'IDsw':635, 'FPS':9.997820328723746},
                        'vanilla_downsampling':     {'MOTA':0.329, 'IDsw':651, 'FPS':11.996064715800753},
                        'skip1':                    {'MOTA':0.307, 'IDsw':570, 'FPS':16.56956741670473},
                        'skip1_downsampling':       {'MOTA':0.300, 'IDsw':577, 'FPS':20.99549490626117},
                        'skip1_prob':               {'MOTA':0.310, 'IDsw':594, 'FPS':17.757431631657468},
                        'skip1_downsampling_prob':  {'MOTA':0.304, 'IDsw':601, 'FPS':19.84908451201911},
                        'skip2':                    {'MOTA':0.263, 'IDsw':581, 'FPS':20.68914912740483},
                        'skip2_downsampling':       {'MOTA':0.256, 'IDsw':538, 'FPS':25.16089209909558},
                        'skip2_prob':               {'MOTA':0.282, 'IDsw':581, 'FPS':21.671446463250184},
                        'skip2_downsampling_prob':  {'MOTA':0.272, 'IDsw':561, 'FPS':22.40254902085448},
                        'skip4':                    {'MOTA':0.174, 'IDsw':858, 'FPS':30.685362182060103},
                        'skip4_downsampling':       {'MOTA':0.173, 'IDsw':826, 'FPS':32.71531749075337},
                        'skip4_prob':               {'MOTA':0.220, 'IDsw':778, 'FPS':26.27376224976722},
                        'skip4_downsampling_prob':  {'MOTA':0.215, 'IDsw':721, 'FPS':28.642976056106697},
                        'skip6':                    {'MOTA':0.115, 'IDsw':809, 'FPS':35.487960011851555},
                        'skip6_downsampling':       {'MOTA':0.111, 'IDsw':821, 'FPS':36.79384991372523},
                        'skip6_prob':               {'MOTA':0.177, 'IDsw':758, 'FPS':28.440129230784574},
                        'skip6_downsampling_prob':  {'MOTA':0.172, 'IDsw':760, 'FPS':30.637051555246025},
        },
        
        'Mobilenetv1': {'vanilla':                  {'MOTA':0.190, 'IDsw':577, 'FPS':9.8474409913098},
                        'vanilla_downsampling':     {'MOTA':0.175, 'IDsw':647, 'FPS':10.164599991702229},
                        'skip1':                    {'MOTA':0.177, 'IDsw':520, 'FPS':15.964183221155363},
                        'skip1_downsampling':       {'MOTA':0.166, 'IDsw':569, 'FPS':16.252922740164827},
                        'skip1_prob':               {'MOTA':0.180, 'IDsw':523, 'FPS':14.404656840037314},
                        'skip1_downsampling_prob':  {'MOTA':0.168, 'IDsw':573, 'FPS':14.878176973130104},
                        'skip2':                    {'MOTA':0.157, 'IDsw':494, 'FPS':20.658319625464184},
                        'skip2_downsampling':       {'MOTA':0.144, 'IDsw':530, 'FPS':20.67430430123686},
                        'skip2_prob':               {'MOTA':0.165, 'IDsw':503, 'FPS':17.894475042242103},
                        'skip2_downsampling_prob':  {'MOTA':0.155, 'IDsw':553, 'FPS':18.656240404675874},
                        'skip4':                    {'MOTA':0.107, 'IDsw':569, 'FPS':27.779953587538593},
                        'skip4_downsampling':       {'MOTA':0.095, 'IDsw':557, 'FPS':28.811689015745316},
                        'skip4_prob':               {'MOTA':0.132, 'IDsw':552, 'FPS':22.686678543601083},
                        'skip4_downsampling_prob':  {'MOTA':0.120, 'IDsw':567, 'FPS':24.149181226030407},
                        'skip6':                    {'MOTA':0.077, 'IDsw':586, 'FPS':32.62759230159879},
                        'skip6_downsampling':       {'MOTA':0.060, 'IDsw':632, 'FPS':33.97791236616167},
                        'skip6_prob':               {'MOTA':0.107, 'IDsw':558, 'FPS':26.286393126809276},
                        'skip6_downsampling_prob':  {'MOTA':0.095, 'IDsw':604, 'FPS':27.845049291724436},
        },
        
        'squeezenetv1_0':   {'vanilla':                 {'MOTA':0.099, 'IDsw':484, 'FPS':20.042197668270926},
                            'vanilla_downsampling':     {'MOTA':0.094, 'IDsw':433, 'FPS':20.717160451288965},
                            'skip1':                    {'MOTA':0.093, 'IDsw':470, 'FPS':33.892409156740605},
                            'skip1_downsampling':       {'MOTA':0.089, 'IDsw':438, 'FPS':34.70319714140534},
                            'skip1_prob':               {'MOTA':0.094, 'IDsw':475, 'FPS':32.83395018961477},
                            'skip1_downsampling_prob':  {'MOTA':0.090, 'IDsw':442, 'FPS':34.2523215195093},
                            'skip2':                    {'MOTA':0.084, 'IDsw':408, 'FPS':42.13578772760607},
                            'skip2_downsampling':       {'MOTA':0.082, 'IDsw':372, 'FPS':43.91007535351279},
                            'skip2_prob':               {'MOTA':0.086, 'IDsw':412, 'FPS':40.69885878710128},
                            'skip2_downsampling_prob':  {'MOTA':0.083, 'IDsw':368, 'FPS':43.13519724065963},
                            'skip4':                    {'MOTA':0.068, 'IDsw':328, 'FPS':53.7290647954817},
                            'skip4_downsampling':       {'MOTA':0.066, 'IDsw':301, 'FPS':56.63252656732362},
                            'skip4_prob':               {'MOTA':0.073, 'IDsw':343, 'FPS':49.940943534497414},
                            'skip4_downsampling_prob':  {'MOTA':0.069, 'IDsw':312, 'FPS':53.98874989205801},
                            'skip6':                    {'MOTA':0.056, 'IDsw':301, 'FPS':62.0070361055159},
                            'skip6_downsampling':       {'MOTA':0.054, 'IDsw':284, 'FPS':63.02879991725403},
                            'skip6_prob':               {'MOTA':0.061, 'IDsw':313, 'FPS':58.15515176744455},
                            'skip6_downsampling_prob':  {'MOTA':0.060, 'IDsw':297, 'FPS':59.56392827961034},
        },
    }

    mot_eval = MOT_eval(data)
    mot_eval.visualization()