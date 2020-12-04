#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'Model Asset Exchange Server'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'max-nested-named-entity-tagger'
    assert metadata['name'] == 'MAX Nested Named Entity Tagger'
    assert metadata['description'] == 'Named Entity Recognition model trained on Genia dataset'
    assert metadata['license'] == 'Mozilla Public 2.0'

def test_response():
    model_endpoint = 'http://localhost:5000/model/predict'
    text = 'The peri-kappa B site mediates human-immunodeficiency virus type 2 enhancer activation, in monocytes but not in T cells.' # noqa
    test_json = {
        "text": text
    }
    expected_response = {
                        "status": "ok",
                        "predictions": {
                            "entities": [
                                    "O",
                                    "B-G#DNA_domain_or_region",
                                    "I-G#DNA_domain_or_region",
                                    "L-G#DNA_domain_or_region",
                                    "O",
                                    "B-G#other_name|B-G#DNA_domain_or_region|B-G#virus",
                                    "I-G#other_name|I-G#DNA_domain_or_region|I-G#virus",
                                    "I-G#other_name|I-G#DNA_domain_or_region|I-G#virus",
                                    "I-G#other_name|I-G#DNA_domain_or_region|I-G#virus",
                                    "I-G#other_name|I-G#DNA_domain_or_region|L-G#virus",
                                    "L-G#DNA_domain_or_region",
                                    "U-G#other_name",
                                    "O",
                                    "U-G#cell_type",
                                    "O",
                                    "O",
                                    "O",
                                    "B-G#cell_type",
                                    "L-G#cell_type",
                                    "O"],
                            "input_terms": [
                                "The",
                                "peri-kappa",
                                "B",
                                "site",
                                "mediates",
                                "human-",
                                "immunodeficiency",
                                "virus",
                                "type",
                                "2",
                                "enhancer",
                                "activation",
                                "in",
                                "monocytes",
                                "but",
                                "not",
                                "in",
                                "T",
                                "cells",
                                "."]
                            }
                        }
    r = requests.post(url=model_endpoint, json=test_json)

    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'
    assert expected_response == response


if __name__ == '__main__':
    pytest.main([__file__])
