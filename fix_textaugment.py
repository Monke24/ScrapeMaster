# Create a fake module to satisfy the import
import sys
import types

# Create the missing module
mod = types.ModuleType('textblob.translate')
sys.modules['textblob.translate'] = mod

# Add the missing class
class NotTranslated(Exception):
    pass

mod.NotTranslated = NotTranslated