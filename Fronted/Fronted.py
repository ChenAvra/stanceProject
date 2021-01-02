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
    def run_btn(self,model1,model2,model3,set_1,set_2,set_3,set_4):
        dataSet = -1
        models = []
        if set_1.active:
            dataSet=1

        elif set_2.active:
            dataSet=2
        elif set_3.active:
            dataSet=3
        elif set_4.active:
            dataSet=4
        if model1.active:
            models.append(1)
            self.manager.get_screen("stat_window").ids.button1.disabled = False
        if model2.active:
            models.append(2)
            self.manager.get_screen("stat_window").ids.button2.disabled = False
        if model3.active:
            models.append(3)
            self.manager.get_screen("stat_window").ids.button3.disabled = False
        print("models: ", models)
        print("dataset: ", dataSet)
        if (not dataSet==-1) and (not len(models)==0):
            set_1.active=False
            set_2.active = False
            set_3.active = False
            set_4.active = False
            model1.active=False
            model2.active = False
            model3.active = False
            if not 1 in models:
                self.manager.get_screen("stat_window").ids.button1.disabled = True
            if not 2 in models:
                self.manager.get_screen("stat_window").ids.button2.disabled = True
            if not 3 in models:
                self.manager.get_screen("stat_window").ids.button3.disabled = True

            self.manager.current = 'stat_window'


class StatWindow(Screen,GridLayout):
    def select_model(self,model):
        self.manager.get_screen("model_stat_window").entry(model)
        self.manager.current = 'model_stat_window'


    def select_dataset(self):
        self.manager.current = 'dataset_stat_window'

class ModelStatWindow(Screen,GridLayout):


    def entry(self,model):
        print(model)
        accuracy = 75.543
        matrix_path = "WhatsApp Image 2021-01-02 at 19.12.21.jpeg"
        classification_report = ""

        self.ids.title.text = "Model " + str(model) + " Stat"
        self.ids.accuracy.text = "Accuracy : " + str(accuracy)




class DataSetStatWindow(Screen,GridLayout):
    pass

class UserStanceWindow(Screen,GridLayout):
    topic = ObjectProperty(None)
    sentence = ObjectProperty(None)

    def test(self):
        dropdown = DropDown()
        for index in range(10):
            # Adding button in drop down list
            btn = Button(text='Value % d' % index, size_hint_y=None, height=40)

            # binding the button to show the text when selected
            btn.bind(on_release=lambda btn: dropdown.select(btn.text))

            # then add the button inside the dropdown
            dropdown.add_widget(btn)

        dropdown.bind(on_select=lambda instance, x: setattr(self.ids.topic, 'text', x))
        dropdown.open(self.ids.topic)

    def apply_btn(self):
        if self.topic.text!="" and self.sentence.text:
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