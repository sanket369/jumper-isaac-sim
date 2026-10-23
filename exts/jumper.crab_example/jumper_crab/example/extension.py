# Registers the crab example in Isaac Sim's example browser (Examples > Policy > Crab).

import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate

from .crab_example import CrabExample


class CrabExampleExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Crab"
        self.category = "Policy"

        overview = (
            "The Jumper crab robot walking on its policy trained in MuJoCo (jumper.posture), "
            "driven by the robot's own controller: the bundle's controller.so builds the "
            "observation and switches modes exactly as it does on the robot."
        )
        overview += "\n\nPress LOAD, then click the viewport so it receives the keyboard:"
        overview += "\n\tW / S or Up / Down: walk forward / back"
        overview += "\n\tA / D or Left / Right: step left / right"
        overview += "\n\tJ / L: turn left / right      H / ;: twist"
        overview += "\n\tI / K: nose down / up      U / O: roll      N / M (held): taller / lower"
        overview += "\n\tEsc: let go of everything"
        overview += "\n\tCtrl + 1..4: dances      Ctrl: leave a dance or gesture"
        overview += "\n\tC: camera follows the robot on/off"
        overview += "\n\nA held key ramps its command to full over 2 s. The full key list "
        overview += "is printed to the console on LOAD, from the bundle's own manual."

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "Crab: MuJoCo policy on the robot's controller",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_policy_deployment.html",
            "overview": overview,
            "sample": CrabExample(),
        }
        ui_handle = BaseSampleUITemplate(**ui_kwargs)
        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)
