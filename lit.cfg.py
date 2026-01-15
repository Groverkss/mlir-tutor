import os
import lit.formats
from lit.llvm import llvm_config

config.name = "Tutorial"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.join(os.path.dirname(__file__), "tutorial")
config.test_exec_root = os.path.join(config.tutorial_obj_root, "test")

# Add tools directories to PATH
llvm_config.with_environment("PATH", config.tutorial_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
