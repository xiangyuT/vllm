from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import tarfile

import crcmod

root = Path(__file__).resolve().parent
destination = Path('/home/intel/LLM/MiniCPM-V-4.7-35B-A3B-DSpark-1008')
partial = destination / 'MiniCPM-V-4.7-35B-A3B-DSpark-1008.tar.partial'
final = destination / 'MiniCPM-V-4.7-35B-A3B-DSpark-1008.tar'
url = 'https://data-transfer-huawei.obs.cn-north-4.myhuaweicloud.com/caitianchi/v47/1006/release/MiniCPM-V-4.7-35B-A3B-DSpark.tar'
assert partial.is_file() and not final.exists()
assert partial.stat().st_size == 778465280
assert not Path(str(partial) + '.aria2').exists()
digest = hashlib.sha256()
crc = crcmod.Crc(0x142F0E1EBA9EA3693, initCrc=0, rev=True, xorOut=0xffffffffffffffff)
with partial.open('rb') as handle:
    while data := handle.read(16 * 1024**2):
        digest.update(data)
        crc.update(data)
assert crc.crcValue == 666953002590505572, 'CRC64 does not match OBS'
print('Complete-object CRC64 matches OBS', flush=True)
with tarfile.open(partial, 'r:') as archive:
    members = archive.getmembers()
    seen = set()
    roots = set()
    payload_bytes = 0
    for member in members:
        p = PurePosixPath(member.name)
        assert not p.is_absolute() and '..' not in p.parts
        assert member.isfile() or member.isdir()
        assert member.sparse is None
        assert str(p) not in seen
        seen.add(str(p)); roots.add(p.parts[0])
        payload_bytes += member.size if member.isfile() else 0
    assert len(roots) == 1
    prefix = next(iter(roots)) + '/'
    for member in members:
        if member.isfile():
            assert not (destination / member.name.removeprefix(prefix)).exists()
    assert shutil.disk_usage(destination).free > payload_bytes + 1024**3
    (root / 'archive-members.json').write_text(json.dumps([
        {'path': member.name, 'bytes': member.size, 'directory': member.isdir()}
        for member in members], indent=2) + '\n')
partial.rename(final)
subprocess.run(['tar', '--extract', '--file', str(final), '--directory', str(destination),
                '--strip-components=1', '--no-same-owner', '--no-same-permissions'], check=True)
configs = {str(p.relative_to(destination)): json.loads(p.read_text())
           for p in destination.rglob('*config.json')}
files = sorted(destination.rglob('*.safetensors'))
headers = {}
dtype_counts = Counter()
sizes = {'BOOL':1,'U8':1,'I8':1,'U16':2,'I16':2,'BF16':2,'F16':2,'U32':4,'I32':4,
         'F32':4,'F64':8,'I64':8,'U64':8,'F8_E4M3':1,'F8_E5M2':1}
total_tensor_bytes = 0
for p in files:
    with p.open('rb') as handle:
        length = struct.unpack('<Q',handle.read(8))[0]
        assert 0 < length < min(p.stat().st_size,64*1024**2)
        header = json.loads(handle.read(length))
    ranges = []
    for name,spec in header.items():
        if name == '__metadata__':
            continue
        assert name not in headers
        begin,end = spec['data_offsets']
        assert 0 <= begin <= end <= p.stat().st_size-8-length
        assert end-begin == math.prod(spec['shape'])*sizes[spec['dtype']]
        headers[name] = {'shard':str(p.relative_to(destination)),**spec}
        dtype_counts[spec['dtype']] += 1
        ranges.append((begin,end)); total_tensor_bytes += end-begin
    cursor = 0
    for begin,end in sorted(ranges):
        assert begin == cursor
        cursor = end
    assert cursor == p.stat().st_size-8-length
index_files = list(destination.rglob('*.safetensors.index.json'))
for p in index_files:
    index = json.loads(p.read_text())
    assert set(index['weight_map']) == set(headers)
    assert all(headers[name]['shard'] == shard for name,shard in index['weight_map'].items())
    assert total_tensor_bytes == index['metadata']['total_size']
(root / 'checkpoint-configs.json').write_text(json.dumps(configs,indent=2) + '\n')
(root / 'safetensors-headers.json').write_text(json.dumps(headers,indent=2) + '\n')
report = {'source_url':url,'local_label':'1008','model_path':str(destination),
          'archive_path':str(final),'archive_bytes':final.stat().st_size,
          'sha256':digest.hexdigest(),'crc64':crc.crcValue,'source_crc64':666953002590505572,
          'source_etag':'4c4fc75d4e7b334a484245ee08a65184-8',
          'source_last_modified':'2026-10-08T06:10:50Z','crc64_matches':True,
          'archive_entries':len(members),'payload_bytes':payload_bytes,
          'safetensors_shards':[str(p.relative_to(destination)) for p in files],
          'tensor_count':len(headers),'tensor_bytes':total_tensor_bytes,'dtype_counts':dict(dtype_counts),
          'gpu_execution':False,'completed_at':datetime.now(timezone.utc).isoformat()}
(root / 'download-manifest.json').write_text(json.dumps(report,indent=2) + '\n')
(destination / 'DOWNLOAD_MANIFEST_1008.json').write_text(json.dumps(report,indent=2) + '\n')
(destination / 'SOURCE_URL.txt').write_text(url + '\n')
(destination / 'ARCHIVE_SHA256.txt').write_text(digest.hexdigest() + '  ' + final.name + '\n')
(destination / 'ARCHIVE_CRC64.txt').write_text(str(crc.crcValue) + '\n')
print(json.dumps(report,indent=2),flush=True)
