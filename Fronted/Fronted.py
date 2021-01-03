from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import ObjectProperty, StringProperty, DictProperty, ListProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.graphics import *
from kivy.animation import Animation,AnimationTransition

from kivy.clock import Clock
from kivy.properties import ListProperty
from kivy.core.window import Window
from kivy.uix.image import Image

Builder.load_file("Fronted.kv")

class WelcomeWindow(Screen,GridLayout):
    def btn(self):
        print("hi")


class SelectWindow(Screen,GridLayout):

    def info_btn(self,index):
        if index=="dataset1":
            self.manager.get_screen("dataset_stat_window").entry("1",True)
            self.manager.current = 'dataset_stat_window'
        elif index=="dataset2":
            self.manager.get_screen("dataset_stat_window").entry("2", True)
            self.manager.current = 'dataset_stat_window'
        elif index=="dataset3":
            self.manager.get_screen("dataset_stat_window").entry("3", True)
            self.manager.current = 'dataset_stat_window'
        elif index=="dataset4":
            self.manager.get_screen("dataset_stat_window").entry("4", True)
            self.manager.current = 'dataset_stat_window'

        else:
            text = ""
            if index=="method1":
                text="Method 1 Information:"
            elif index=="method2":
                text="This algorithm was created by UCL Machine  Reading (UCLMR) \n\n during Stage 1 of the Fake News Challenge (FNC-1) in 2017.\n\n It is based on a single, end-to-end system consisting of lexical \n\n as well as similarity features passed through a multi-layer \n\n perceptron with one hidden layer. UCLMR won third place in \n\n the FNC however out of the three best scoring teams, \n\n UCLMR’s classifier is the simplest and easiest to understand."
            elif index=="method3":
                text="TAN - Target-specific Attention Neural Network. This method \n\n consists of two main components: a recurrent neural network (RNN) \n\n as the feature extractor for text and a fully-connected network \n\n as the target-specific attention selector. It’s a special mechanism \n\n which drives the model to concentrate on salient parts in text \n\n with respect to a specific target. \n\n This algorithm is based on LSTM (similar to RNN). \n\n ** Note that running this algorithm takes a long time \ndue to its complexity."


            close_button = Button(text="close", size_hint=(None, None), size=(475, 50))
            layout = GridLayout(cols=1)
            layout.add_widget(Label(text=text ))
            layout.add_widget(close_button)
            popup = Popup(title='Info', content=layout, size_hint=(None, None), size=(500, 500))
            popup.open()
            close_button.bind(on_press=popup.dismiss)

    def run_btn(self,model1,model2,model3,set_1,set_2,set_3,set_4,percent):
        dataSet = -1
        models = []
        datasetNumber=-1
        if set_1.active:
            dataSet='semEval2016'
            datasetNumber=1
        elif set_2.active:
            dataSet='FNC-1'
            datasetNumber=2
        elif set_3.active:
            dataSet='MPCHI'
            datasetNumber=3
        elif set_4.active:
            dataSet='EmergentLite'
            datasetNumber=4
        if model1.active:
            models.append(1)
        if model2.active:
            models.append(2)
        if model3.active:
            models.append(3)
        if (not dataSet==-1) and (not len(models)==0 and percent.text.isnumeric() and int(percent.text)>0)and int(percent.text)<100:
            set_1.active=False
            set_2.active = False
            set_3.active = False
            set_4.active = False
            model1.active=False
            model2.active = False
            model3.active = False


            self.manager.get_screen("stat_window").entry(models, dataSet, datasetNumber)
            self.manager.current = 'stat_window'


class StatWindow(Screen,GridLayout):
    # models = []
    # dataset = StringProperty('default')
    def entry(self, models, dataSet,datasetNumber):
        self.models = models
        self.dataset= str(dataSet)
        if 1 in models:
            self.ids.button1.disabled = False
            self.ids.method_1_accuracy.text = "75% Accuracy"
        else:
            self.ids.button1.disabled = True
        if 2 in models:
            self.ids.button2.disabled = False
            self.ids.method_2_accuracy.text = "45% Accuracy"
        else:
            self.ids.button2.disabled = True
        if 3 in models:
            self.ids.button3.disabled = False
            self.ids.method_3_accuracy.text = "80% Accuracy"
        else:
            self.ids.button3.disabled = True
        self.ids.dataset_button.text = "{} info".format(dataSet)
        self.manager.get_screen("dataset_stat_window").entry(str(datasetNumber))

    def select_model(self,model):
        self.manager.get_screen("model_stat_window").entry(model)
        self.manager.current = 'model_stat_window'




    def select_dataset(self):
        self.manager.current = 'dataset_stat_window'


class ModelStatWindow(Screen,GridLayout):


    def entry(self,model):
        accuracy = 75.543
        matrix_path = "WhatsApp Image 2021-01-02 at 19.12.21.jpeg"
        against_precision = 0.53
        against_recall = 0.47
        against_f = 0.5
        against_support = 49
        favor_precision = 0.52
        favor_recall = 0.72
        favor_f = 0.6
        favor_support = 71
        none_precision = 0.56
        none_recall = 0.38
        none_f = 0.45
        none_support = 66
        wavg_precision = 0.54
        wavg_recall = 0.53
        wavg_f = 0.52
        wavg_support = 186

        self.ids.title.text = "Model " + str(model) + " Stat"
        self.ids.accuracy.text = "Accuracy : " + str(accuracy)
        self.ids.matrix.source = matrix_path
        self.ids.ap.text = str(against_precision)
        self.ids.ar.text = str(against_recall)
        self.ids.af.text = str(against_f)
        self.ids.aq.text = str(against_support)
        self.ids.fp.text = str(favor_precision)
        self.ids.fr.text = str(favor_recall)
        self.ids.ff.text = str(favor_f)
        self.ids.fq.text = str(favor_support)
        self.ids.np.text = str(none_precision)
        self.ids.nr.text = str(none_recall)
        self.ids.nf.text = str(none_f)
        self.ids.nq.text = str(none_support)
        self.ids.tp.text = str(wavg_precision)
        self.ids.tr.text = str(wavg_recall)
        self.ids.tf.text = str(wavg_f)
        self.ids.tq.text = str(wavg_support)





class DataSetStatWindow(Screen,GridLayout):
    def entry(self, dataset, info_button=False):
        if info_button:
            self.ids.back_stat.disabled = True
            self.ids.back_select.disabled = False
        else:
            self.ids.back_stat.disabled = False
            self.ids.back_select.disabled = True

        if dataset=="1":
            self.ids.title.text = "semEval2016 info"
            self.ids.info.text = "This dataset was provided at the SemEval competition in 2016. The data provided contains instances of: tweets, id, target, and stance,\n\n where stance is one of  the following: for, against, none. The dataset contains 4,042 records."
            self.ids.dataset_photo.source= 'semEval2016.png'
        elif dataset=="2":
            self.ids.title.text = "FNC-1 info"
            self.ids.info.text = "This dataset was provided at the Fake News Chalenge (FNC-1) in 2017. The data provided contains instances of: headline, body and stance,\n\n where stance is one of  the following: unrelated, discuss, agree, disagree. The dataset contains 75,385 records."
            self.ids.dataset_photo.source= 'FNC.png'
        elif dataset=="3":
            self.ids.title.text = "MPCHI info"
            self.ids.info.text = "This dataset contains health-related online news articles. The data provided contains instances of: tweets, id, target and stance,\n\n where stance is one of  the following: favor, against, none. The dataset contains 1,533 records."
            self.ids.dataset_photo.source= 'mpchi.png'
        elif dataset=="4":
            self.ids.title.text = "EmergentLite info"
            self.ids.info.text = "This dataset contains claims extracted from rumour sites and Twitter, with 300 claims and 2,595 headlines.\n\n The stance is one of the following: for, against, observing."
            self.ids.dataset_photo.source= 'emergent.png'


class UserStanceWindow(Screen,GridLayout):
    topic = ObjectProperty(None)
    sentence = ObjectProperty(None)

    def test(self):
        dropdown = DropDown()
        for index in ["E-ciggarettes are safer than normal ciggarettes","Sun exposure can lead to skin cancer","Vitamin C prevents common cold","Women should take HRT post menopause","MMR vaccine can cause autism","atheism","hillary clinton","legalization of abortion","climate change is a real concern","feminist movement"]:
            # Adding button in drop down list
            btn = Button(text=index, size_hint_y=None, height=40)

            # binding the button to show the text when selected
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))

            # then add the button inside the dropdown
            dropdown.add_widget(btn)

        dropdown.bind(on_select=lambda instance, x: setattr(self.ids.topic, 'text', x))
        dropdown.open(self.ids.topic)

    def apply_btn(self):
        if self.topic.text!="Topic" and self.sentence.text:
            close_button = Button(text="close")
            layout = GridLayout(cols=1)
            layout.add_widget(Label(text='Topic:  ' + self.topic.text))
            layout.add_widget(Label(text='Sentence:  ' + self.sentence.text))
            layout.add_widget(Label(text='Stance:  Supported'))
            layout.add_widget(Label(text=''))
            layout.add_widget(close_button)
            popup = Popup(title='Reveal Your Stance', content=layout, size_hint=(None, None), size=(500, 500))
            popup.open()
            close_button.bind(on_press=popup.dismiss)
            self.sentence.text = ""


class Manager(ScreenManager):
    page = None


class TestApp(App):
    def build(self):
        manager = Manager()
        manager.add_widget(WelcomeWindow(name='welcome'))
        manager.add_widget(UserStanceWindow(name='user_stance'))
        manager.add_widget(SelectWindow(name='select_window'))
        manager.add_widget(StatWindow(name='stat_window'))
        manager.add_widget(ModelStatWindow(name='model_stat_window'))
        manager.add_widget(DataSetStatWindow(name='dataset_stat_window'))
        return manager

if __name__ == '__main__':
    TestApp().run()