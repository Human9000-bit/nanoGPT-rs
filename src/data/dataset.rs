use burn::data::dataset::{Dataset, HuggingfaceDatasetLoader, SqliteDataset};
use derive_new::new;
use serde::{Deserialize, Serialize};

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DbPediaItem {
    pub content: String,
}

/// The dbppedia_14 dataset
///
/// https://huggingface.co/datasets/dbpedia_14
pub struct DbPediaDataset {
    dataset: SqliteDataset<DbPediaItem>,
}

impl Dataset<TextGenerationItem> for DbPediaDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem::new(item.content))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DbPediaDataset {
    /// Constructs a new dataset from the given split
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("dbpedia_14")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }

    /// Alias for dataset with split "train"
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Alias for dataset with split "test"
    pub fn test() -> Self {
        Self::new("test")
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DvachItem {
    pub conversation: Vec<String>,
}

/// The 2ch.hk dataset
///
/// https://huggingface.co/datasets/Vikhrmodels/2ch-24-09-2024-no-links
pub struct DvachDataset {
    pub dataset: SqliteDataset<DvachItem>,
}

impl DvachDataset {
    /// Constructs a new dataset from the given split
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DvachItem> =
            HuggingfaceDatasetLoader::new("Vikhrmodels/2ch-24-09-2024-no-links")
                .dataset(split)
                .unwrap();

        Self { dataset }
    }

    /// Alias for train dataset
    pub fn train() -> Self {
        Self::new("train")
    }
}

impl Dataset<TextGenerationItem> for DvachDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem::new(item.conversation.join(" ")))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
