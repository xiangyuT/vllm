import json
import os
from pathlib import Path
import uuid

import torch

plan = json.loads(Path('/workspace/evidence/PLAN.json').read_text())
assert os.environ['ZE_AFFINITY_MASK'] == '0,1'
assert torch.xpu.device_count() == 2
devices = []
for index, expected in enumerate(plan['expected_sysman_uuid']):
    props = torch.xpu.get_device_properties(index)
    sysman_uuid = str(uuid.UUID(bytes=uuid.UUID(str(props.uuid)).bytes[::-1]))
    assert sysman_uuid == expected, (index, sysman_uuid, expected)
    assert props.device_id == 0xE223
    devices.append({'logical': index, 'torch_uuid': str(props.uuid),
                    'sysman_uuid': sysman_uuid, 'pci': plan['expected_pci'][index],
                    'name': props.name, 'device_id': props.device_id,
                    'total_memory': props.total_memory})
print(json.dumps({'affinity': os.environ['ZE_AFFINITY_MASK'],
                  'torch_version': torch.__version__, 'devices': devices}, indent=2))
