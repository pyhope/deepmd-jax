from deepmd_jax.resume import executable_signature_digest

def test_alias_topology_is_not_part_of_metadata_key():
    shared = tuple([17, 33, 65])
    a = dict(shapes=[shared, shared], dtype='float32')
    b = dict(shapes=[tuple([17,33,65]),tuple([17,33,65])], dtype='float32')
    assert executable_signature_digest(a)==executable_signature_digest(b)
    b['dtype']='float16'
    assert executable_signature_digest(a)!=executable_signature_digest(b)
